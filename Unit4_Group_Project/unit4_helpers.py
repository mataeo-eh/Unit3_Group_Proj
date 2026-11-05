"""Unit 4 helpers: two-GPU frozen DINOv3 extraction and imbalance-aware heads.

How this file fits the project
------------------------------
Unit 3's ``../baselines.py`` remains the single source of truth for the shared
data contract: ``split_dataset`` (seed-42 stratified 70/15/15 split),
``preprocess_image`` (native resolution, ImageNet mean/std), ``WarmStableDecay``
(the 10% warmup / 70% stable / 20% decay learning-rate schedule), and
``evaluate_probabilities`` (accuracy, macro P/R/F1, per-class metrics, ROC/PR).
``build_notebooks.py`` embeds that module first and this module second, so every
name used below from Unit 3 is already defined when these cells run on Kaggle.

What this module adds for Unit 4
--------------------------------
* ``make_two_gpu_strategy``: a ``tf.distribute.MirroredStrategy`` over both
  Kaggle T4 GPUs; it refuses to run on any other GPU count.
* ``extract_split_features``: batched backbone inference spread across both
  GPUs, followed by the same 2x2 "area" spatial grid used by the Unit 3
  spatial experiments (12x16x768 feature map -> 2x2x768 -> 3,072 vector).
* Logit-output heads plus three losses: plain cross entropy (control),
  alpha-balanced focal loss (experiment 1), and logit-adjusted cross entropy
  (experiment 2, Menon et al., ICLR 2021).
* ``train_head_two_gpu``: trains for the full WSD budget, saving both the best
  validation-loss checkpoint and the last-epoch weights.
* Reporting helpers for best-vs-last comparisons on train/validation/test.

All modeling is TensorFlow / Keras 3. No other ML framework is used.
"""
import json
import math
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# DINOv3 ConvNeXt-Tiny's last stage has 768 channels. The Unit 3 spatial
# experiments reduced the (H', W', 768) map to a 2x2 grid with area resizing.
FEATURE_CHANNELS = 768
SPATIAL_GRID = (2, 2)


# ---------------------------------------------------------------------------
# Distribution strategy
# ---------------------------------------------------------------------------
def make_two_gpu_strategy(expected_gpus=2):
    """Return a MirroredStrategy that spans exactly ``expected_gpus`` GPUs.

    MirroredStrategy keeps one replica of every variable per GPU and splits
    each global batch evenly between replicas. HierarchicalCopyAllReduce is the
    documented all-reduce that works without NVLink, which the Kaggle T4 pair
    lacks. Failing fast here prevents a silent single-GPU or CPU run.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) != expected_gpus:
        raise RuntimeError(
            f'Expected {expected_gpus} GPUs (Kaggle "GPU T4 x2"); found {len(gpus)}: {gpus}')
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    if strategy.num_replicas_in_sync != expected_gpus:
        raise RuntimeError(f'Strategy has {strategy.num_replicas_in_sync} replicas, not {expected_gpus}.')
    print('MirroredStrategy replicas:', strategy.num_replicas_in_sync,
          '| devices:', strategy.extended.worker_devices)
    return strategy


def _single_worker_options():
    """tf.data options for single-machine MirroredStrategy input pipelines.

    With one worker, Keras splits each global batch across the local GPUs; no
    file/data sharding between workers is needed, so auto-sharding is OFF
    (this also avoids the "cannot shard" fallback warning for in-memory data).
    """
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    return options


# ---------------------------------------------------------------------------
# Frozen backbone extraction on both GPUs
# ---------------------------------------------------------------------------
def load_frozen_backbone(model_file, strategy):
    """Load the saved DINOv3 ConvNeXt-Tiny under the strategy scope, frozen.

    Keras 3 records the active strategy when a model is created, so the model
    must be loaded inside ``strategy.scope()`` for ``predict`` to use both GPUs.
    The saved model's output is a dict with two documented keys:
      * ``pooled``      (N, 768)          global average of the last stage
      * ``feature_map`` (N, H', W', 768)  last-stage spatial map (stride 32)
    Both are kept: the training notebook uses ``feature_map``; the feature
    dataset notebook stores both.
    """
    with strategy.scope():
        backbone = keras.models.load_model(model_file, compile=False)
        backbone.trainable = False
    if not isinstance(backbone.output, dict) or set(backbone.output) != {'pooled', 'feature_map'}:
        raise ValueError('Expected the Tiny backbone pooled/feature_map output contract.')
    assert not backbone.trainable_weights
    return backbone


def native_shape_groups(split):
    """Group row indices by native (height, width) so batches need no padding.

    TrashNet's resized release is uniformly 384x512, giving one group, but the
    grouping keeps the "no resize, crop, or padding" rule true for any input.
    Images are decoded through the Hugging Face ``Image`` feature, exactly as
    in Unit 3 (it applies EXIF orientation before we read the size).
    """
    groups = {}
    for index, row in enumerate(split):
        width, height = row['image'].size  # PIL reports (width, height)
        groups.setdefault((height, width), []).append(index)
    return groups


def _pixel_dataset(split, indices, height, width, global_batch):
    """Batched tf.data of normalized pixels for rows ``indices`` (all one shape).

    Each element is produced by Unit 3's ``preprocess_image`` so pixel values are
    bit-identical to the Unit 3 per-image pipeline; only the batching differs.
    """
    subset = split.select(indices)

    def generate():
        for row in subset:
            yield preprocess_image(row['image'])[0].numpy()

    data = tf.data.Dataset.from_generator(
        generate, output_signature=tf.TensorSpec((height, width, 3), tf.float32))
    return data.batch(global_batch).with_options(_single_worker_options()).prefetch(2)


def spatial_grid_vectors(feature_maps, grid=SPATIAL_GRID, chunk=256):
    """Area-resize (N, H', W', C) maps to ``grid`` and flatten row-major.

    This is the same operation as the Unit 3 ``spatial_2x2_vector`` cell
    (``tf.image.resize(..., method='area')`` then reshape), applied in chunks.
    For the 12x16 map and a 2x2 grid, each output cell is the mean of a 6x8
    block, so the 3,072-vector keeps coarse top/bottom/left/right layout.
    """
    feature_maps = np.asarray(feature_maps)
    n, channels = len(feature_maps), feature_maps.shape[-1]
    vectors = np.empty((n, grid[0] * grid[1] * channels), dtype=np.float32)
    for start in range(0, n, chunk):
        block = tf.convert_to_tensor(feature_maps[start:start + chunk], dtype=tf.float32)
        resized = tf.image.resize(block, size=grid, method='area')
        vectors[start:start + len(block)] = tf.reshape(resized, (len(block), -1)).numpy()
    return vectors


def extract_split_features(backbone, split, split_name, strategy, per_replica_batch=8):
    """Run the frozen backbone once per image of ``split`` using both GPUs.

    Returns a dict with:
      * ``pooled``      float32 (N, 768)
      * ``feature_map`` float32 (N, H', W', 768)   (requires one native shape)
      * ``labels``      int32   (N,)
      * ``native_shapes`` {"(H, W)": count}
    Rows keep the split's original order. ``predict`` concatenates per-replica
    outputs in batch order; ``check_batched_parity`` verifies this explicitly.
    """
    global_batch = per_replica_batch * strategy.num_replicas_in_sync
    groups = native_shape_groups(split)
    if len(groups) != 1:
        # A single dense (N, H', W', 768) array needs one map size. TrashNet's
        # resized release satisfies this; fail loudly instead of padding.
        raise ValueError(f'{split_name}: multiple native sizes {list(groups)}; '
                         'store per-shape arrays before using this helper.')
    (height, width), indices = next(iter(groups.items()))
    data = _pixel_dataset(split, indices, height, width, global_batch)
    outputs = backbone.predict(data, verbose=0)
    pooled = np.asarray(outputs['pooled'], dtype=np.float32)
    feature_map = np.asarray(outputs['feature_map'], dtype=np.float32)
    labels = np.asarray(split['label'], dtype=np.int32)
    assert pooled.shape == (len(split), FEATURE_CHANNELS), pooled.shape
    assert feature_map.shape[0] == len(split) and feature_map.shape[-1] == FEATURE_CHANNELS
    if not (np.isfinite(pooled).all() and np.isfinite(feature_map).all()):
        raise FloatingPointError(f'{split_name}: non-finite backbone features')
    shapes = {str((height, width)): len(indices)}
    print(f'{split_name}: {len(split)} images | native {shapes} | map {feature_map.shape[1:]}', flush=True)
    return {'pooled': pooled, 'feature_map': feature_map, 'labels': labels, 'native_shapes': shapes}


def check_batched_parity(backbone, split, feature_map, count=16, tolerance=1e-3):
    """Compare batched two-GPU maps with Unit 3-style single-image inference.

    The first ``count`` rows span both replicas of the first global batch, so a
    replica-ordering mistake would produce O(1) errors. Small differences are
    expected because cuDNN may choose different kernels for batch 1 vs batch 8.
    The check uses the max absolute error relative to the reference's max
    magnitude, and raises if it exceeds ``tolerance``.
    """
    worst = 0.0
    for index in range(min(count, len(split))):
        pixels = preprocess_image(split[index]['image'])
        reference = backbone(pixels, training=False)['feature_map'].numpy()[0]
        error = np.max(np.abs(reference - feature_map[index])) / max(np.max(np.abs(reference)), 1e-12)
        worst = max(worst, float(error))
    print(f'Batched vs single-image relative max error over {count} rows: {worst:.2e}')
    if worst > tolerance:
        raise AssertionError(f'Batched extraction differs from single-image inference ({worst:.2e}).')
    return worst


# ---------------------------------------------------------------------------
# Heads, losses, and data pipelines for imbalance experiments
# ---------------------------------------------------------------------------
def build_logit_head(feature_dim, num_classes, hidden_units=(40, 24), dropout_rates=(0.1, 0.1), name='head'):
    """Unit 3's "larger ANN" architecture, but returning raw logits.

    Architecture (same as Unit 3 ``larger_ann``): Dense(40, GELU) -> Dropout(0.1)
    -> Dense(24, GELU) -> Dropout(0.1) -> Dense(num_classes). Logits (rather
    than softmax probabilities) let every loss use ``from_logits=True`` and let
    the logit-adjusted loss add its per-class offsets before the softmax.
    Probabilities for evaluation come from ``predict_probabilities``.
    """
    if len(dropout_rates) > len(hidden_units):
        raise ValueError('Dropout rates must correspond to existing hidden layers.')
    inputs = keras.Input(shape=(feature_dim,), name='features')
    x = inputs
    for i, units in enumerate(hidden_units):
        x = keras.layers.Dense(units, activation='gelu', name=f'hidden_{i}')(x)
        if i < len(dropout_rates):
            x = keras.layers.Dropout(dropout_rates[i], name=f'dropout_{i}')(x)
    logits = keras.layers.Dense(num_classes, name='logits')(x)
    return keras.Model(inputs, logits, name=name)


def balanced_class_weights(labels, num_classes):
    """Inverse-frequency weights N / (K * n_c) (sklearn's "balanced" formula).

    Their sample-weighted mean is exactly 1, so the loss scale stays comparable
    to unweighted cross entropy. Keras' CategoricalFocalCrossentropy docstring
    recommends this formula for a per-class ``alpha`` in multi-class problems.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError('Every class needs training examples to define a weight.')
    return counts.sum() / (num_classes * counts)


@keras.saving.register_keras_serializable(package='Unit4')
class LogitAdjustedCrossentropy(keras.losses.Loss):
    """Logit-adjusted softmax cross entropy (Menon et al., ICLR 2021).

    Loss: CE(y, softmax(f(x) + tau * log(pi))), where pi are training class
    priors. Adding log-priors only during training forces the raw logits f(x)
    to win by a larger margin on rare classes; at inference the raw logits are
    used unchanged. With tau = 1 this equals Balanced Softmax (Ren et al. 2020)
    and is Fisher-consistent for the *balanced* error (mean per-class error).
    Optional label smoothing is applied to the one-hot targets, as in Keras'
    built-in categorical cross entropy.

    Expects one-hot ``y_true`` and logit ``y_pred`` of shape (batch, classes).
    """

    def __init__(self, class_priors, tau=1.0, label_smoothing=0.0,
                 name='logit_adjusted_crossentropy', reduction='sum_over_batch_size', dtype=None):
        super().__init__(name=name, reduction=reduction, dtype=dtype)
        priors = np.asarray(class_priors, dtype=np.float64)
        if (priors <= 0).any():
            raise ValueError('Class priors must be positive.')
        self.class_priors = [float(p) for p in priors / priors.sum()]
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)
        self._offsets = (self.tau * np.log(np.asarray(self.class_priors))).astype(np.float32)

    def call(self, y_true, y_pred):
        offsets = keras.ops.convert_to_tensor(self._offsets, dtype=y_pred.dtype)
        return keras.losses.categorical_crossentropy(
            y_true, y_pred + offsets, from_logits=True, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super().get_config()
        config.update({'class_priors': self.class_priors, 'tau': self.tau,
                       'label_smoothing': self.label_smoothing})
        return config


def make_loss(spec, train_labels, num_classes):
    """Build the loss named by ``spec['loss']`` from training labels only.

    * ``crossentropy``    plain softmax CE (control; Unit 3's loss).
    * ``focal``           alpha-balanced focal loss (Lin et al. 2017): Keras'
                          CategoricalFocalCrossentropy with per-class alpha from
                          ``balanced_class_weights`` and focusing ``gamma``.
    * ``logit_adjusted``  ``LogitAdjustedCrossentropy`` with training priors.
    Validation/test labels never influence weights or priors.
    """
    kind = spec['loss']
    if kind == 'crossentropy':
        return keras.losses.CategoricalCrossentropy(from_logits=True)
    if kind == 'focal':
        alpha = [float(a) for a in balanced_class_weights(train_labels, num_classes)]
        return keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=spec['gamma'], from_logits=True)
    if kind == 'logit_adjusted':
        counts = np.bincount(train_labels, minlength=num_classes)
        return LogitAdjustedCrossentropy(counts / counts.sum(), tau=spec['tau'],
                                         label_smoothing=spec.get('label_smoothing', 0.0))
    raise ValueError(f'Unknown loss: {kind}')


def make_optimizer(spec, schedule):
    """Adam (Unit 3 default) or AdamW when ``spec`` sets ``weight_decay``.

    AdamW applies decoupled weight decay. Biases are excluded so decay acts
    only on weight matrices, matching common practice and the classifier
    "weight balancing" motivation (Alshammari et al., CVPR 2022).
    """
    weight_decay = spec.get('weight_decay')
    if not weight_decay:
        return keras.optimizers.Adam(learning_rate=schedule)
    optimizer = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=weight_decay)
    optimizer.exclude_from_weight_decay(var_names=['bias'])
    return optimizer


def onehot_feature_dataset(x, y, num_classes, global_batch, training=False, seed=42):
    """In-memory features with one-hot labels (required by focal/CE-from-one-hot).

    Training shuffles every epoch with a fixed seed. ``global_batch`` is split
    evenly across the GPUs by MirroredStrategy (e.g. 256 -> 128 per T4).
    """
    labels = tf.one_hot(np.asarray(y, dtype=np.int32), num_classes, dtype=tf.float32)
    data = tf.data.Dataset.from_tensor_slices((np.asarray(x, dtype=np.float32), labels))
    if training:
        data = data.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    return data.batch(global_batch).with_options(_single_worker_options()).prefetch(2)


def train_head_two_gpu(spec, cache, class_names, strategy, output_dir, *, batch_size, epochs, peak_lr, seed=42):
    """Train one head on both GPUs for the full WSD budget; keep best and last.

    Unlike Unit 3, there is no early stopping: Unit 3 runs stopped at epochs
    11-27 of 50, before the 80% decay boundary, so the WSD "decay" phase never
    ran. Training the whole schedule makes the last-epoch head the annealed
    WSD endpoint, which is the meaningful counterpart to the best-validation
    checkpoint. ModelCheckpoint keeps the strictly-best ``val_loss`` epoch.

    Returns (best_model, last_model, metadata). Both models are reloaded from
    disk with ``compile=False`` so they are plain single-device inference models.
    """
    keras.utils.set_random_seed(seed)
    num_classes = len(class_names)
    x, y = cache['train']
    vx, vy = cache['validation']
    steps_per_epoch = math.ceil(len(y) / batch_size)
    total_steps = epochs * steps_per_epoch
    output_dir = Path(output_dir)
    best_path = output_dir / f"{spec['name']}_best.keras"
    last_path = output_dir / f"{spec['name']}_last.keras"

    with strategy.scope():
        model = build_logit_head(x.shape[1], num_classes, name=spec['name'])
        schedule = WarmStableDecay(total_steps, peak_lr)  # from Unit 3 baselines.py
        model.compile(
            optimizer=make_optimizer(spec, schedule),
            loss=make_loss(spec, y, num_classes),
            metrics=[keras.metrics.CategoricalAccuracy(name='accuracy'),
                     # threshold=None -> argmax prediction, so this is macro F1.
                     keras.metrics.F1Score(average='macro', threshold=None, name='f1_macro')])

    history = model.fit(
        onehot_feature_dataset(x, y, num_classes, batch_size, training=True, seed=seed),
        validation_data=onehot_feature_dataset(vx, vy, num_classes, batch_size),
        epochs=epochs, verbose=2,
        callbacks=[keras.callbacks.ModelCheckpoint(best_path, monitor='val_loss', mode='min', save_best_only=True),
                   keras.callbacks.CSVLogger(output_dir / f"{spec['name']}_epochs.csv")])
    model.save(last_path)

    records = {k: [float(v) for v in values] for k, values in history.history.items()}
    # Learning rate at the last optimizer step of each epoch, from the schedule itself.
    records['learning_rate_end_of_epoch'] = [
        float(schedule((epoch + 1) * steps_per_epoch - 1)) for epoch in range(epochs)]
    metadata = {'name': spec['name'], 'spec': spec, 'parameters': model.count_params(),
                'epochs_run': len(records['loss']), 'steps_per_epoch': steps_per_epoch,
                'total_steps': total_steps, 'global_batch': batch_size,
                'per_replica_batch': batch_size // strategy.num_replicas_in_sync,
                'peak_learning_rate': peak_lr,
                'best_epoch': int(np.argmin(records['val_loss'])) + 1, 'history': records}
    (output_dir / f"{spec['name']}_history.json").write_text(json.dumps(metadata, indent=2))
    best = keras.models.load_model(best_path, compile=False)
    last = keras.models.load_model(last_path, compile=False)
    return best, last, metadata


def predict_probabilities(model, x, batch_size=256):
    """Softmax of a logit head's predictions, as float32 probabilities."""
    logits = model.predict(np.asarray(x, dtype=np.float32), batch_size=batch_size, verbose=0)
    return tf.nn.softmax(tf.convert_to_tensor(logits, tf.float32), axis=-1).numpy()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize_report(report, class_names):
    """Flatten an ``evaluate_probabilities`` report into one comparison row.

    Adds class-equity measures: macro recall is balanced accuracy; worst-class
    F1 and the F1 standard deviation show how evenly classes are served.
    ``loss`` is plain cross entropy on probabilities for every model, so it is
    comparable across models even though their training losses differ.
    """
    f1 = np.array([row['f1'] for row in report['per_class']])
    row = {key: report[key] for key in ('accuracy', 'f1_macro', 'recall_macro', 'precision_macro',
                                        'pr_auc_macro', 'roc_auc_macro')}
    row.update({'balanced_accuracy': report['recall_macro'], 'worst_class_f1': float(f1.min()),
                'worst_class': class_names[int(f1.argmin())], 'f1_std': float(f1.std()),
                'ce_loss': report['loss']})
    for item in report['per_class']:
        row[f"f1_{item['class']}"] = item['f1']
    return row


def plot_training_curves(histories, output_dir):
    """Loss, validation accuracy/macro-F1, and learning rate for each run."""
    for name, meta in histories.items():
        h = meta['history']
        epochs = np.arange(1, meta['epochs_run'] + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(epochs, h['loss'], label='Train')
        axes[0].plot(epochs, h['val_loss'], label='Validation')
        axes[0].set(title=f'{name}: training-loss value', xlabel='Epoch', ylabel=meta['spec']['loss'])
        axes[1].plot(epochs, h['accuracy'], label='Train accuracy')
        axes[1].plot(epochs, h['val_accuracy'], label='Val accuracy')
        axes[1].plot(epochs, h['val_f1_macro'], label='Val macro F1')
        axes[1].set(title='Accuracy / macro F1', xlabel='Epoch', ylim=(0.8, 1.005))
        axes[2].plot(epochs, h['learning_rate_end_of_epoch'])
        axes[2].set(title='WSD learning rate (end of epoch)', xlabel='Epoch')
        for ax in axes[:2]:
            ax.axvline(meta['best_epoch'], color='grey', linestyle='--', label='Best val_loss')
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f'{name}_learning_curves.png')
        plt.show()


def plot_confusion_grid(results, class_names, split, output_dir):
    """One confusion matrix per (model, checkpoint) for ``split``."""
    keys = list(results)
    cols = 2
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        cm = np.asarray(results[key][split]['confusion_matrix'])
        ax.imshow(cm, cmap='Blues')
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
               xticklabels=class_names, yticklabels=class_names,
               xlabel='Predicted', ylabel='True', title=f'{key} — {split}')
        ax.tick_params(axis='x', rotation=45)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
    for ax in list(axes.flat)[len(keys):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f'confusion_matrices_{split}.png')
    plt.show()


def plot_per_class_f1(comparison, class_names, split, output_dir):
    """Grouped bars of per-class F1 for every (model, checkpoint) on ``split``."""
    subset = comparison[comparison['split'] == split]
    width = 0.8 / len(subset)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, (_, row) in enumerate(subset.iterrows()):
        values = [row[f'f1_{c}'] for c in class_names]
        ax.bar(np.arange(len(class_names)) + k * width, values, width, label=row['model'])
    ax.set(xticks=np.arange(len(class_names)) + 0.4 - width / 2, xticklabels=class_names,
           ylabel='F1', ylim=(0.5, 1.0), title=f'Per-class F1 — {split}')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f'per_class_f1_{split}.png')
    plt.show()
