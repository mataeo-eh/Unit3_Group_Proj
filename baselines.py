"""Reusable TensorFlow heads, frozen-feature caching, and train/dev reporting.

The notebook embeds this module so Kaggle needs only Notebook.ipynb. No helper
accepts a test feature array: test evaluation belongs to a later final-model run.
"""
import json
import math
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import DatasetDict


def split_dataset(published, seed=42, test_fraction=0.15, validation_fraction=0.15):
    """Honor DatasetDict author splits; stratify only missing partitions.

    Dataset.train_test_split returns DatasetDict with 'train' and 'test' keys.
    Its temporary 'test' partition becomes validation when splitting train/dev.
    Existing columns are preserved; row_id records original split membership.
    """
    splits = DatasetDict({name: data.add_column('row_id', [f'{name}:{i}' for i in range(len(data))])
                          for name, data in published.items()})
    if 'train' not in splits:
        raise ValueError('A published train split is required.')
    if 'validation' not in splits and 'dev' in splits:
        splits['validation'] = splits.pop('dev')
    if 'test' not in splits:
        partition = splits['train'].train_test_split(
            test_size=test_fraction, seed=seed, stratify_by_column='label')
        splits['train'], splits['test'] = partition['train'], partition['test']
        dev_fraction = validation_fraction / (1 - test_fraction)
    else:
        dev_fraction = validation_fraction
    if 'validation' not in splits:
        partition = splits['train'].train_test_split(
            test_size=dev_fraction, seed=seed, stratify_by_column='label')
        splits['train'], splits['validation'] = partition['train'], partition['test']
    ids = {name: set(splits[name]['row_id']) for name in ('train', 'validation', 'test')}
    assert not (ids['train'] & ids['validation'] or ids['train'] & ids['test'] or ids['validation'] & ids['test'])
    return splits


def preprocess_image(rgb_image):
    """Preserve every native RGB pixel; apply only scale and ImageNet normalization."""
    pixels = tf.convert_to_tensor(np.asarray(rgb_image.convert('RGB')), dtype=tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    return ((pixels - mean) / std)[None, ...]


def prepare_pooled_backbone(backbone, mixed_precision=False):
    """Build a frozen pooled-only graph, preserving every pretrained weight.

    Eager Functional execution retains intermediate tensors until the call ends.
    At native multi-megapixel resolutions the expanded ConvNeXt activations can
    exhaust a T4 even with batch size one. Graph execution enables tensor-lifetime
    planning; pooled-only output also prunes the unused spatial-output branch.
    On GPU, mixed-float16 convolution/dense activations further halve their size.
    Normalization and global pooling stay float32; all variables stay float32.
    Heads and the shared RAM cache remain float32. No spatial pixels are changed.
    """
    if not isinstance(backbone.output, dict) or set(backbone.output) != {'pooled', 'feature_map'}:
        raise ValueError('Expected the Tiny backbone pooled/feature_map output contract.')
    pooled_model = keras.Model(backbone.inputs, backbone.output['pooled'], name='frozen_pooled_backbone')
    if mixed_precision:
        def clone_layer(layer):
            # Documented clone_function receives each non-Input layer instance.
            # Its config is copied explicitly; no checkpoint weights are modified.
            config = layer.get_config()
            config['dtype'] = 'float32' if isinstance(layer, (keras.layers.LayerNormalization, keras.layers.GlobalAveragePooling2D)) else 'mixed_float16'
            return layer.__class__.from_config(config)
        mixed_model = keras.models.clone_model(pooled_model, clone_function=clone_layer)
        mixed_model.set_weights(pooled_model.get_weights())
        pooled_model = mixed_model
    pooled_model.trainable = False
    assert not pooled_model.trainable_weights
    return pooled_model


def pooled_inference_function(pooled_model):
    """One dynamic-shape graph for all native image sizes, returning float32."""
    @tf.function(input_signature=[tf.TensorSpec((1, None, None, 3), tf.float32)])
    def infer(pixels):
        return tf.cast(pooled_model(pixels, training=False), tf.float32)
    return infer


def cache_pooled_features(backbone, splits):
    """One extraction pass, one inference per train/dev image, shared by all heads.

    The prepared frozen model returns only pooled (N,768), pruning spatial output.
    Single-image inference supports mixed native sizes without padding or resize.
    Dataset rows supply image, label, row_id; other columns are unused.
    """
    backbone.trainable = False
    assert not backbone.trainable_weights
    infer = pooled_inference_function(backbone)
    cache, shapes, calls = {}, {}, 0
    for name in ('train', 'validation'):
        data = splits[name]
        vectors = np.empty((len(data), 768), dtype=np.float32)
        labels = np.asarray(data['label'], dtype=np.int32)
        for i, row in enumerate(data):
            pixels = preprocess_image(row['image'])
            pooled = infer(pixels)
            assert tuple(pooled.shape) == (1, 768)
            tf.debugging.assert_all_finite(pooled, 'Non-finite pooled features')
            vectors[i] = pooled.numpy()[0]
            shape = str(tuple(pixels.shape[1:3]))
            shapes[shape] = shapes.get(shape, 0) + 1
            calls += 1
            if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(data):
                print(f'{name}: cached {i + 1}/{len(data)}', flush=True)
        cache[name] = (vectors, labels)
    assert calls == len(splits['train']) + len(splits['validation'])
    print('Backbone inference calls:', calls, '| Native resolutions:', shapes)
    print('Feature cache MiB:', sum(x.nbytes + y.nbytes for x, y in cache.values()) / 2**20)
    return cache, {'inference_calls': calls, 'native_shapes': shapes, 'test_inference_calls': 0}


@keras.saving.register_keras_serializable(package='Unit3')
class WarmStableDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Batch-based linear 10% warmup / 70% stable / 20% decay over 50 epochs.

    Early stopping may finish before decay; the schedule uses the planned budget,
    never the unknowable eventual stopping epoch. End learning rate is 1% of peak.
    """
    def __init__(self, total_steps, peak_lr=0.001):
        self.total_steps = int(total_steps)
        self.peak_lr = float(peak_lr)
        self.warmup = max(1, int(round(total_steps * 0.1)))
        self.decay_start = int(round(total_steps * 0.8))

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.peak_lr * (step + 1) / self.warmup
        progress = tf.clip_by_value((step - self.decay_start) / max(1, self.total_steps - self.decay_start - 1), 0., 1.)
        decay = self.peak_lr * (1. - 0.99 * progress)
        return tf.where(step < self.warmup, warm, tf.where(step < self.decay_start, self.peak_lr, decay))

    def get_config(self):
        return {'total_steps': self.total_steps, 'peak_lr': self.peak_lr}


def build_head(feature_dim, num_classes, hidden_units=(), name='linear', dropout_rates=()):
    """Dense softmax classifier for any fixed-width feature representation.

    A spatial experiment can supply vectors of another width or add its own
    spatial aggregation before this head. The default is multinomial logistic
    regression; hidden_units=(8,) adds one small GELU layer. Optional dropout
    rates apply after the corresponding hidden layers, only during training.
    """
    if len(dropout_rates) > len(hidden_units):
        raise ValueError('Dropout rates must correspond to existing hidden layers.')
    inputs = keras.Input(shape=(feature_dim,), name='features')
    x = inputs
    for i, units in enumerate(hidden_units):
        x = keras.layers.Dense(units, activation='gelu', name=f'hidden_{i}')(x)
        if i < len(dropout_rates):
            # Keras fit enables dropout; validation/predict keep every unit active.
            x = keras.layers.Dropout(dropout_rates[i], name=f'dropout_{i}')(x)
    probabilities = keras.layers.Dense(num_classes, activation='softmax', name='probabilities')(x)
    return keras.Model(inputs, probabilities, name=name)


def feature_dataset(x, y, batch_size, training=False, seed=42):
    """Batch RAM features with a bounded TF thread pool for portable execution."""
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        data = data.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return data.batch(batch_size).with_options(options).prefetch(1)


def train_head(spec, cache, num_classes, output_dir, batch_size=64, epochs=50, patience=5, seed=42):
    """Train only cached vectors, save strict best val_loss, reload best checkpoint.

    History.history contains every completed epoch including patience epochs.
    CSVLogger persists those rows even when EarlyStopping restores best weights.
    """
    keras.utils.set_random_seed(seed)
    x, y = cache['train']
    dev_x, dev_y = cache['validation']
    model = build_head(x.shape[1], num_classes, spec['hidden_units'], spec['name'],
                       dropout_rates=spec.get('dropout_rates', ()))
    schedule = WarmStableDecay(epochs * math.ceil(len(y) / batch_size), spec['learning_rate']) if spec['wsd'] else spec['learning_rate']
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=schedule),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    checkpoint = Path(output_dir) / f"{spec['name']}_best.keras"
    history = model.fit(feature_dataset(x, y, batch_size, True, seed),
        validation_data=feature_dataset(dev_x, dev_y, batch_size), epochs=epochs,
        callbacks=[keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_loss', mode='min', save_best_only=True),
                   keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0., patience=patience, restore_best_weights=True),
                   keras.callbacks.CSVLogger(Path(output_dir) / f"{spec['name']}_epochs.csv")], verbose=2)
    # Reloading verifies the on-disk deliverable and makes final scoring explicit.
    best = keras.models.load_model(checkpoint, compile=False)
    records = {k: [float(v) for v in values] for k, values in history.history.items()}
    metadata = {'name': spec['name'], 'parameters': best.count_params(), 'epochs_run': len(records['loss']),
                'best_epoch': int(np.argmin(records['val_loss'])) + 1,
                'stopped_early': len(records['loss']) < epochs, 'history': records, 'spec': spec}
    (Path(output_dir) / f"{spec['name']}_history.json").write_text(json.dumps(metadata, indent=2))
    return best, metadata


def evaluate_probabilities(labels, probabilities, class_names):
    """Argmax classification metrics plus one-vs-rest ROC/PR for each class.

    Keras AUC's documented TP/FP/TN/FN state yields curves at 2000 thresholds.
    Macro precision/recall/F1 gives each class equal weight; zero denominators
    produce 0. Cross entropy uses Keras epsilon clipping, including the one-hot
    majority baseline, whose wrong-class ideal loss would otherwise be infinite.
    """
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    assert probabilities.shape == (len(labels), len(class_names))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-5)
    predictions = probabilities.argmax(axis=1)
    cm = tf.math.confusion_matrix(labels, predictions, num_classes=len(class_names)).numpy()
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(len(tp)), where=cm.sum(axis=0) != 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(len(tp)), where=cm.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(tp)), where=precision + recall != 0)
    per_class, curves = [], {}
    for i, name in enumerate(class_names):
        truth = (labels == i).astype(np.int32)
        if truth.min() == truth.max():
            raise ValueError(f'{name} has no positive or no negative examples; AUC is undefined.')
        roc = keras.metrics.AUC(curve='ROC', num_thresholds=2000)
        pr = keras.metrics.AUC(curve='PR', num_thresholds=2000)
        roc.update_state(truth, probabilities[:, i])
        pr.update_state(truth, probabilities[:, i])
        # These four public metric variables are counts per threshold, not guesses.
        t, f, n, missed = [v.numpy() for v in (roc.true_positives, roc.false_positives, roc.true_negatives, roc.false_negatives)]
        curves[name] = {'fpr': np.divide(f, f+n, out=np.zeros_like(f), where=f+n != 0).tolist(),
                        'tpr': np.divide(t, t+missed, out=np.zeros_like(t), where=t+missed != 0).tolist(),
                        'precision': np.divide(t, t+f, out=np.ones_like(t), where=t+f != 0).tolist()}
        per_class.append({'class': name, 'support': int(cm[i].sum()), 'precision': float(precision[i]),
                          'recall': float(recall[i]), 'f1': float(f1[i]),
                          'roc_auc': float(roc.result()), 'pr_auc': float(pr.result())})
    return {'accuracy': float(np.mean(predictions == labels)),
            'loss': float(keras.losses.SparseCategoricalCrossentropy()(labels, probabilities)),
            'precision_macro': float(precision.mean()), 'recall_macro': float(recall.mean()), 'f1_macro': float(f1.mean()),
            'pr_auc_macro': float(np.mean([r['pr_auc'] for r in per_class])),
            'roc_auc_macro': float(np.mean([r['roc_auc'] for r in per_class])),
            'confusion_matrix': cm.tolist(), 'per_class': per_class, 'curves': curves}


def plot_reports(results, histories, class_names, output_dir):
    """Plot every registered run; adding future runs needs no plotting rewrite."""
    for name, history in histories.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        epochs = np.arange(1, history['epochs_run'] + 1)
        ax.plot(epochs, history['history']['loss'], label='Train')
        ax.plot(epochs, history['history']['val_loss'], label='Validation')
        ax.axvline(history['best_epoch'], color='grey', linestyle='--', label='Best checkpoint')
        ax.set(title=f'{name}: all completed epochs', xlabel='Epoch', ylabel='Cross entropy')
        ax.legend(); fig.tight_layout(); fig.savefig(Path(output_dir) / f'{name}_learning_curve.png'); plt.show()
    for split in ('train', 'validation'):
        for name, reports in results.items():
            report = reports[split]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            cm = np.asarray(report['confusion_matrix'])
            axes[0].imshow(cm, cmap='Blues')
            axes[0].set(xticks=range(len(class_names)), yticks=range(len(class_names)),
                        xticklabels=class_names, yticklabels=class_names, xlabel='Predicted', ylabel='True', title='Confusion matrix')
            axes[0].tick_params(axis='x', rotation=45)
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    axes[0].text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
            for label, curve in report['curves'].items():
                axes[1].plot(curve['fpr'], curve['tpr'], label=label)
                axes[2].plot(curve['tpr'], curve['precision'], label=label)
            axes[1].plot([0, 1], [0, 1], 'k--')
            axes[1].set(xlabel='False positive rate', ylabel='Recall', title='One-vs-rest ROC')
            axes[2].set(xlabel='Recall', ylabel='Precision', title='One-vs-rest PR')
            axes[1].legend(fontsize=8); axes[2].legend(fontsize=8)
            fig.suptitle(f'{name} — {split}'); fig.tight_layout()
            fig.savefig(Path(output_dir) / f'{name}_{split}_metrics.png'); plt.show()
