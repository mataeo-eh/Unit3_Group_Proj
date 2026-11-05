"""Generate the two standalone Unit 4 Kaggle notebooks.

Run from anywhere after editing ``unit4_helpers.py`` or ``../baselines.py``:

    python build_notebooks.py

Outputs (each folder is one Kaggle kernel; push with ``kaggle kernels push -p <folder>``):
  * ``training/Unit4_Training.ipynb``          control + experiment 1 + experiment 2
  * ``feature_extraction/Unit4_Feature_Extraction.ipynb``  builds the reusable
    frozen-feature dataset so later notebooks can skip the backbone entirely.

Both notebooks embed Unit 3's ``baselines.py`` (split, preprocessing, WSD
schedule, metrics) followed by ``unit4_helpers.py``, so Kaggle needs only the
.ipynb file plus the attached dataset and model inputs.
"""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
BASELINES_SOURCE = (ROOT.parent / 'baselines.py').read_text()
UNIT4_SOURCE = (ROOT / 'unit4_helpers.py').read_text()

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

# ---------------------------------------------------------------------------
# Cells shared by both notebooks
# ---------------------------------------------------------------------------
DEPENDENCY_MD = '''## 1. Compatible dependencies

Kaggle's November 2026 GPU image ships TensorFlow 2.21, protobuf 7.36, Keras
3.13.2, and datasets 4.8.5. TensorFlow and protobuf are kept as shipped; Unit 3's
TF 2.20 / protobuf 5.29 downgrades now conflict with preinstalled Google Cloud
packages. Only these are changed:
- `keras==3.15.1`: the version that saved and validated the DINOv3 artifact.
- `datasets==5.1.0`: the version that produced the Unit 3 split. Its stratified
  split code matches 4.8.5, but its image-folder loader differs, so row order
  could differ.
- `Pillow==12.3.0`: the Unit 3 JPEG decoder version; it also satisfies Gradio's `pillow<13`.
- `google-cloud-bigquery-storage==2.43.0`: required by preinstalled BigFrames.
- `ydf==0.17.0`: the shipped 0.15 requires `protobuf<7`.
- Uninstalled, all unused: Dopamine, Google Colab, and legacy MoviePy (pin
  conflicts), plus RAPIDS `cudf-cu13`, `dask-cudf-cu13`, and `cuml-cu13`. cudf
  26.6 requires `pyarrow<24`, while datasets 5.1 requires `pyarrow>=24`.

`pip check` must pass before any import, so a dependency conflict fails setup
instead of being ignored. The environment was verified with a probe kernel.
'''

DEPENDENCY_CODE = '''# Install compatible pins before importing TensorFlow or dataset libraries.
import subprocess
import sys
import time

# Runtime covers setup, extraction, training, scoring, and plots (not queue time).
notebook_started = time.perf_counter()

# Unused preinstalled distributions whose requirements conflict (see markdown above).
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y',
                'dopamine-rl', 'google-colab', 'moviepy',
                'cudf-cu13', 'dask-cudf-cu13', 'cuml-cu13'], check=True)
requirements = [
    'keras==3.15.1', 'datasets==5.1.0', 'Pillow==12.3.0',
    'google-cloud-bigquery-storage==2.43.0', 'ydf==0.17.0',
]
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                '--disable-pip-version-check', *requirements], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)

from importlib import metadata
print({name: metadata.version(name) for name in
       ('tensorflow', 'keras', 'protobuf', 'datasets', 'pyarrow', 'Pillow', 'kagglehub')})
'''

CONFIG_IMPORTS = '''import os
from pathlib import Path

# Select the backend before importing Keras, and reduce TensorFlow log noise.
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import pandas as pd
import tensorflow as tf
# Memory growth must be configured before any GPU tensor or strategy exists.
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import keras
import kagglehub
from datasets import ClassLabel, DatasetDict, Image, load_dataset
from IPython.display import display

print("TensorFlow:", tf.__version__, "| Keras:", keras.__version__)
print("Visible GPUs:", tf.config.list_physical_devices('GPU'))

# Data and model inputs (attached to the Kaggle kernel).
DATASET_HANDLE = "feyzazkefe/trashnet"     # Kaggle TrashNet, 2,527 images at 512x384
DATASET_SUBDIR = "dataset-resized"         # one folder per class inside the dataset
MODEL_HANDLE = "mataeoanderson/dinov3-convnext/keras/tiny/1"
MODEL_FILENAME = "dinov3_convnext_tiny.keras"
SEED = 42
# Backbone inference batch per GPU; global batch = 8 x 2 GPUs = 16 images.
PER_REPLICA_EXTRACTION_BATCH = 8
keras.utils.set_random_seed(SEED)
'''

HELPERS_MD = '''## {number}. Embedded helpers

The first cell is Unit 3's `baselines.py`, unchanged: the stratified splitter,
native-resolution preprocessing, the `WarmStableDecay` learning-rate schedule,
and `evaluate_probabilities`. The second cell is `unit4_helpers.py`: two-GPU
extraction, logit heads, the three losses, and reporting. `build_notebooks.py`
regenerates these cells from the source files.
'''

DATASET_MD = '''## {number}. Load TrashNet from the attached Kaggle dataset

`kagglehub.dataset_download` returns the local directory of the attached
dataset (no download from Hugging Face). The Hugging Face `imagefolder` loader
reads the class folders. Each row has `image` (PIL, decoded on access) and
`label` (`ClassLabel` integer). Class names come from the folder names.
'''

DATASET_CODE = '''dataset_root = Path(kagglehub.dataset_download(DATASET_HANDLE))
DATA_DIR = dataset_root / DATASET_SUBDIR
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Expected TrashNet image folders at {DATA_DIR}")

dataset = load_dataset("imagefolder", data_dir=str(DATA_DIR), split="train")
label_feature = dataset.features["label"]
assert isinstance(label_feature, ClassLabel)
class_names = label_feature.names
print(dataset)
print("Classes:", class_names)
assert class_names == ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
'''

SPLIT_MD = '''## {number}. Same stratified 70/15/15 split as Unit 3

`split_dataset` with seed 42 reproduces the partitions of the final Unit 3 run
(train 1,768 / validation 379 / test 380). The counts are asserted against
that run so results stay comparable. The manifest records each row's split,
original row id, label, and image path relative to the dataset folder.
'''

SPLIT_CODE = '''splits = split_dataset(DatasetDict({'train': dataset}), seed=SEED)
counts = pd.DataFrame({name: np.bincount(np.asarray(splits[name]['label']), minlength=len(class_names))
                       for name in ('train', 'validation', 'test')}, index=class_names)
display(counts)

# Per-class counts printed by the final Unit 3 notebook (same seed and loader).
UNIT3_COUNTS = {'train': [282, 351, 286, 416, 338, 95],
                'validation': [60, 75, 62, 89, 72, 21],
                'test': [61, 75, 62, 89, 72, 21]}
for name, expected in UNIT3_COUNTS.items():
    assert counts[name].tolist() == expected, (name, counts[name].tolist(), expected)

def split_manifest_rows(split, split_name):
    """Undecoded image column gives {'path', 'bytes'}; only the path is used.

    TrashNet's layout is <class>/<file>.jpg, so the relative path is the class
    folder plus file name; the folder must agree with the ClassLabel label.
    """
    files = split.cast_column('image', Image(decode=False))['image']
    rows = []
    for index, (file, label, row_id) in enumerate(zip(files, split['label'], split['row_id'])):
        if not file.get('path'):
            raise ValueError(f'{split_name}[{index}] has no source path')
        source = Path(file['path'])
        if source.parent.name != class_names[int(label)]:
            raise ValueError(f'{split_name}[{index}]: folder {source.parent.name} != label {label}')
        rows.append({'split': split_name, 'index': index, 'row_id': row_id, 'label': int(label),
                     'class_name': class_names[int(label)],
                     'relative_path': f'{source.parent.name}/{source.name}'})
    return rows

manifest = pd.DataFrame([row for name in ('train', 'validation', 'test')
                         for row in split_manifest_rows(splits[name], name)])
display(manifest.head())
'''

BACKBONE_CODE = '''strategy = make_two_gpu_strategy(expected_gpus=2)

model_dir = Path(kagglehub.model_download(MODEL_HANDLE))
model_file = model_dir / MODEL_FILENAME
if not model_file.is_file():
    raise FileNotFoundError(f"Expected model artifact: {MODEL_FILENAME}")
backbone = load_frozen_backbone(model_file, strategy)
print("Backbone parameters:", f"{backbone.count_params():,}", "| trainable:", len(backbone.trainable_weights))

extraction_started = time.perf_counter()
raw_features = {}
for name in ('train', 'validation', 'test'):
    raw_features[name] = extract_split_features(
        backbone, splits[name], name, strategy, per_replica_batch=PER_REPLICA_EXTRACTION_BATCH)
extraction_seconds = time.perf_counter() - extraction_started
print(f"Two-GPU extraction time: {extraction_seconds:.1f} s for {len(dataset)} images")

# Batched two-GPU output must match Unit 3-style single-image inference.
parity_error = check_batched_parity(backbone, splits['train'], raw_features['train']['feature_map'])
'''


def embedded_helper_cells(number):
    return [md(HELPERS_MD.format(number=number)), code(BASELINES_SOURCE), code(UNIT4_SOURCE)]


# ---------------------------------------------------------------------------
# Training notebook
# ---------------------------------------------------------------------------
TRAINING_TITLE = '''# Unit 4 group project: TrashNet class-imbalance experiments

Unit 3 showed that **spatially aware features** (the DINOv3 ConvNeXt-Tiny
feature map reduced to a 2x2 grid, 3,072 values) beat global pooled features,
and that the weakest classes were the minority **trash** class (95 of 1,768
training images, 5.4%) and **plastic**, which is confused with trash and glass.
This notebook keeps the best representation, uses the larger prediction head,
and targets the imbalance problem directly:

| Run | Loss | Optimizer | Purpose |
|---|---|---|---|
| `control_ce` | softmax cross entropy | Adam | same setup, Unit 3 loss: isolates the loss change |
| `exp1_focal` | alpha-balanced focal loss (gamma 2, inverse-frequency alpha) | Adam | Experiment 1 |
| `exp2_logit_adjusted` | logit-adjusted cross entropy (tau 1) + label smoothing 0.1 | AdamW (wd 0.05) | Experiment 2 |

Shared by every run: frozen backbone (both T4 GPUs for extraction), 2x2 spatial
features, larger head (40 GELU -> dropout 0.1 -> 24 GELU -> dropout 0.1 -> 6),
global batch 256 (128 per GPU via `MirroredStrategy`), WSD learning rate
(10% warmup / 70% stable / 20% decay to 1% of a 2e-3 peak), 100 epochs. Each
run saves its **best-validation-loss** head and its **last-epoch** head; both
are scored on train, validation, and the held-out test split.

All modeling uses TensorFlow / Keras.
'''

EXTRACTION_TRAIN_MD = '''## {number}. Frozen backbone on both GPUs, then 2x2 spatial features

The backbone is loaded inside `MirroredStrategy.scope()`, so `predict` splits
each 16-image batch into 8 per GPU. All images are decoded and normalized by
Unit 3's `preprocess_image` (native 512x384, no resize/crop/pad). The returned
12x16x768 feature map is area-resized to 2x2 and flattened (3,072 values), the
same operation as the Unit 3 spatial experiments. A parity check compares 16
batched rows with single-image inference. Train, validation, and test are all
extracted here; test features are used only for the final scoring section.
'''

FEATURES_CODE = '''feature_cache = {}
for name, features in raw_features.items():
    feature_cache[name] = (spatial_grid_vectors(features['feature_map']), features['labels'])
    print(name, 'spatial features:', feature_cache[name][0].shape)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
manifest.to_csv(OUTPUT_DIR / 'split_manifest.csv', index=False)
(OUTPUT_DIR / 'feature_extraction.json').write_text(json.dumps({
    'extraction_seconds': extraction_seconds, 'parity_relative_max_error': parity_error,
    'replicas': strategy.num_replicas_in_sync, 'per_replica_batch': PER_REPLICA_EXTRACTION_BATCH,
    'native_shapes': {name: f['native_shapes'] for name, f in raw_features.items()},
    'spatial_grid': list(SPATIAL_GRID), 'feature_dim': int(feature_cache['train'][0].shape[1])}, indent=2))

# Full maps are no longer needed (about 1.5 GB in float32); keep only vectors.
del raw_features
import gc; gc.collect()
'''

DESIGN_MD = '''## {number}. Experiment design

**Why a larger batch is safe without gradient accumulation.** The heads train
on cached 3,072-value vectors, not images. A 256-row batch is 3 MB of
features and the head has about 124k parameters, so GPU memory is not a
constraint and gradient accumulation is unnecessary. 1,768 / 256 gives 7
optimizer steps per epoch; 100 epochs give 700 steps, similar to the step
count where the Unit 3 heads peaked (300-600 steps at batch 64). The peak
learning rate rises from 1e-3 to 2e-3 (square-root scaling for a 4x batch with
Adam).

**Why no early stopping.** In Unit 3 every head stopped at epoch 11-27 of 50,
before the WSD decay phase began at epoch 40. That made "last epoch" just
"five epochs past best". Running the full schedule makes the last-epoch head
the fully annealed WSD result, and `ModelCheckpoint` still saves the
best-validation-loss epoch. Each run logs validation macro F1 every epoch.

**Experiment 1: focal loss.** `keras.losses.CategoricalFocalCrossentropy`
with `gamma = 2` (Lin et al., 2017) down-weights examples the head already
classifies confidently. Per-class `alpha = N / (K * n_c)` (the inverse-frequency
weighting recommended in the Keras docstring) gives trash about 3.1x weight
and paper about 0.7x.

**Experiment 2: logit-adjusted loss with regularization.** Unit 3's remaining
errors are classifier-boundary errors between visually similar materials
(plastic vs trash/glass/metal, cardboard vs paper), with trash as the small
class. The literature on long-tailed recognition with a frozen representation
points to these choices:
- Decoupling (Kang et al., ICLR 2020): with a fixed representation, class
  imbalance mainly biases the classifier, so the fix belongs in the classifier.
  That is exactly our frozen-backbone setup.
- Logit adjustment (Menon et al., ICLR 2021) trains on `logits + tau * log(prior)`.
  This is Fisher-consistent for the balanced error (mean per-class error), the
  "equal performance across classes" goal. With tau = 1 it equals Balanced
  Softmax (Ren et al., NeurIPS 2020). It is the standard strong baseline in
  long-tail work, including recent foundation-model work (Generalized Logit
  Adjustment, 2023; BS-cRT, 2026). It reweights decision boundaries rather
  than examples, so it does not amplify the noisy gradients of the 95 trash
  images the way large per-sample weights can.
- Label smoothing 0.1 improves calibration and curbs over-confidence, which
  matters most in imbalanced settings (MiSLAS, Zhong et al., CVPR 2021). Unit 3
  heads reached about 99.9% train accuracy within a few epochs.
- AdamW weight decay 0.05 on weight matrices: classifier-stage weight decay
  balances per-class weight norms and is one of the most effective long-tail
  fixes (Alshammari et al., CVPR 2022).
- No class-balanced resampling, because combining it with logit adjustment
  over-corrects toward the rare class (Ren et al., 2020).

`control_ce` uses the identical setup with plain cross entropy, so the effect
of each loss can be isolated.
'''

SPECS_CODE = '''HEAD_BATCH_SIZE = 256      # global batch; MirroredStrategy gives 128 per T4
HEAD_EPOCHS = 100          # full WSD budget: 7 steps/epoch -> 700 optimizer steps
PEAK_LEARNING_RATE = 2e-3  # sqrt(256 / 64) x Unit 3's 1e-3

RUN_SPECS = [
    {'name': 'control_ce', 'loss': 'crossentropy'},
    {'name': 'exp1_focal', 'loss': 'focal', 'gamma': 2.0},
    {'name': 'exp2_logit_adjusted', 'loss': 'logit_adjusted', 'tau': 1.0,
     'label_smoothing': 0.1, 'weight_decay': 0.05},
]

train_labels = feature_cache['train'][1]
train_counts = np.bincount(train_labels, minlength=len(class_names))
display(pd.DataFrame({
    'train_count': train_counts,
    'prior': train_counts / train_counts.sum(),
    'focal_alpha (exp1)': balanced_class_weights(train_labels, len(class_names)),
    'logit_offset tau*log(prior) (exp2)': np.log(train_counts / train_counts.sum()),
}, index=class_names).round(4))

# Verify both imbalance losses against explicit NumPy formulas before training.
rng = np.random.default_rng(SEED)
check_logits = rng.normal(size=(8, len(class_names))).astype(np.float32)
check_onehot = np.eye(len(class_names), dtype=np.float32)[rng.integers(0, len(class_names), 8)]
softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
for spec in RUN_SPECS[1:]:
    loss_fn = make_loss(spec, train_labels, len(class_names))
    if spec['loss'] == 'focal':
        alpha = balanced_class_weights(train_labels, len(class_names))
        p = softmax(check_logits)
        expected = np.mean(np.sum(alpha * (1 - p) ** spec['gamma'] * -check_onehot * np.log(p), axis=1))
    else:
        prior = train_counts / train_counts.sum()
        smooth = check_onehot * (1 - spec['label_smoothing']) + spec['label_smoothing'] / len(class_names)
        expected = np.mean(np.sum(-smooth * np.log(softmax(check_logits + spec['tau'] * np.log(prior))), axis=1))
    actual = float(loss_fn(check_onehot, check_logits))
    print(f"{spec['name']}: keras {actual:.6f} | numpy {expected:.6f}")
    assert abs(actual - expected) < 1e-4 * max(1., abs(expected))
'''

TRAIN_CODE = '''trained, histories = {}, {}
for spec in RUN_SPECS:
    print('=' * 80, '\\nTraining:', spec, flush=True)
    started = time.perf_counter()
    best_head, last_head, meta = train_head_two_gpu(
        spec, feature_cache, class_names, strategy, OUTPUT_DIR,
        batch_size=HEAD_BATCH_SIZE, epochs=HEAD_EPOCHS, peak_lr=PEAK_LEARNING_RATE, seed=SEED)
    meta['train_seconds'] = time.perf_counter() - started
    trained[spec['name']] = {'best': best_head, 'last': last_head}
    histories[spec['name']] = meta
    print({k: v for k, v in meta.items() if k != 'history'})
'''

EVAL_MD = '''## {number}. Best-validation vs last-epoch heads on train, validation, and test

Predictions are the argmax of softmax(logits); the training-time logit
offsets of experiment 2 are not applied at inference. `ce_loss` is plain
cross entropy for every model, so it is comparable across losses.
`balanced_accuracy` (= macro recall), `worst_class_f1`, and `f1_std` measure
how evenly classes are served. The Unit 3 reference rows are the selected
spatial logistic-regression head from the final Unit 3 notebook, on the same
splits.
'''

EVAL_CODE = '''results, rows = {}, []
for name, heads in trained.items():
    for checkpoint, head in heads.items():
        key = f'{name}/{checkpoint}'
        results[key] = {}
        for split_name, (x, y) in feature_cache.items():
            probabilities = predict_probabilities(head, x)
            report = evaluate_probabilities(y, probabilities, class_names)
            results[key][split_name] = report
            rows.append({'model': key, 'experiment': name, 'checkpoint': checkpoint,
                         'epoch': histories[name]['best_epoch'] if checkpoint == 'best' else HEAD_EPOCHS,
                         'split': split_name, **summarize_report(report, class_names)})

# Reference values printed by the final Unit 3 notebook (spatial logistic head).
unit3_reference = [
    {'model': 'unit3_spatial_logistic/best', 'experiment': 'unit3_reference', 'checkpoint': 'best',
     'split': 'validation', 'accuracy': 0.9420, 'f1_macro': 0.9282},
    {'model': 'unit3_spatial_logistic/best', 'experiment': 'unit3_reference', 'checkpoint': 'best',
     'split': 'test', 'accuracy': 0.9447, 'f1_macro': 0.9390},
]
comparison = pd.DataFrame(rows)
comparison.to_csv(OUTPUT_DIR / 'comparison.csv', index=False)
(OUTPUT_DIR / 'metrics.json').write_text(json.dumps(results, indent=2))

headline = ['model', 'epoch', 'accuracy', 'f1_macro', 'balanced_accuracy', 'precision_macro',
            'worst_class_f1', 'worst_class', 'f1_std', 'ce_loss', 'pr_auc_macro']
for split_name in ('validation', 'test'):
    print(f'\\n===== {split_name.upper()} =====')
    table = pd.concat([comparison[comparison['split'] == split_name],
                       pd.DataFrame([r for r in unit3_reference if r['split'] == split_name])])
    display(table[headline].sort_values('f1_macro', ascending=False).round(4).reset_index(drop=True))
print('\\n===== TRAIN (diagnostic) =====')
display(comparison[comparison['split'] == 'train'][headline].round(4).reset_index(drop=True))
'''

PER_CLASS_CODE = '''# Per-class test metrics for every head (support, precision, recall, F1, PR-AUC).
per_class_rows = []
for key, reports in results.items():
    for item in reports['test']['per_class']:
        per_class_rows.append({'model': key, **{k: item[k] for k in ('class', 'support', 'precision', 'recall', 'f1', 'pr_auc')}})
per_class = pd.DataFrame(per_class_rows)
per_class.to_csv(OUTPUT_DIR / 'per_class_test.csv', index=False)
display(per_class.pivot(index='model', columns='class', values='f1').round(4))
display(per_class.pivot(index='model', columns='class', values='recall').round(4))
display(per_class.pivot(index='model', columns='class', values='precision').round(4))
'''

PLOTS_MD = '''## {number}. Learning curves, confusion matrices, and per-class F1

Learning curves show every epoch with the best-validation-loss epoch marked,
plus the realized WSD learning rate. Confusion matrices and per-class F1 are
shown for validation and test.
'''

PLOTS_CODE = '''plot_training_curves(histories, OUTPUT_DIR)
for split_name in ('validation', 'test'):
    plot_confusion_grid(results, class_names, split_name, OUTPUT_DIR)
    plot_per_class_f1(comparison, class_names, split_name, OUTPUT_DIR)

elapsed_seconds = time.perf_counter() - notebook_started
(OUTPUT_DIR / 'notebook_runtime.json').write_text(json.dumps(
    {'execution_seconds': elapsed_seconds, 'execution_minutes': elapsed_seconds / 60.,
     'queue_time_included': False}, indent=2))
print('Saved artifacts:', sorted(path.name for path in OUTPUT_DIR.iterdir()))
print(f'Notebook execution time: {elapsed_seconds / 60.:.2f} minutes ({elapsed_seconds:.1f} seconds).')
'''


def training_notebook():
    cells = [md(TRAINING_TITLE), md(DEPENDENCY_MD), code(DEPENDENCY_CODE),
             md('## 2. Configuration and GPU check'), code(CONFIG_IMPORTS)]
    cells += embedded_helper_cells(3)
    cells += [md(DATASET_MD.format(number=4)), code(DATASET_CODE),
              md(SPLIT_MD.format(number=5)), code(SPLIT_CODE),
              md(EXTRACTION_TRAIN_MD.format(number=6)), code(BACKBONE_CODE), code(FEATURES_CODE),
              md(DESIGN_MD.format(number=7)), code(SPECS_CODE), code(TRAIN_CODE),
              md(EVAL_MD.format(number=8)), code(EVAL_CODE), code(PER_CLASS_CODE),
              md(PLOTS_MD.format(number=9)), code(PLOTS_CODE)]
    return cells


# ---------------------------------------------------------------------------
# Feature-extraction (dataset builder) notebook
# ---------------------------------------------------------------------------
EXTRACT_TITLE = '''# Unit 4: TrashNet frozen DINOv3 ConvNeXt-Tiny feature dataset

This notebook runs the frozen backbone **once** over every TrashNet image and
saves its outputs, so later training notebooks can attach this output as a
dataset instead of the images and skip backbone inference entirely.
Augmentation is not used anywhere in the project, so the features are fixed
for a given image.

Saved per split (`train`, `validation`, `test`; same seed-42 stratified 70/15/15
split as Units 3 and 4), in `trashnet_dinov3_convnext_tiny_features/`:

| File | dtype | Shape | Content |
|---|---|---|---|
| `{split}_spatial_2x2.npy` | float32 | (N, 3072) | 2x2 area-pooled feature map (the Unit 3/4 head input) |
| `{split}_pooled.npy` | float32 | (N, 768) | backbone global pooled vector |
| `{split}_feature_map_f16.npy` | float16 | (N, 12, 16, 768) | full last-stage map, for other spatial poolings |
| `{split}_labels.npy` | int32 | (N,) | class ids (0=cardboard ... 5=trash) |
| `metadata.csv` | | | split, index, row_id, label, class name, image path |
| `manifest.json`, `README.md` | | | provenance and loading instructions |

Row *i* of every `{split}_*.npy` file is row *i* of that split in `metadata.csv`.
Both T4 GPUs share inference through `MirroredStrategy`.
'''

EXTRACTION_DATASET_MD = '''## {number}. Frozen backbone on both GPUs

Same extraction path as the Unit 4 training notebook: native 512x384 pixels,
ImageNet normalization, frozen weights, 8 images per GPU per batch, plus a
parity check against single-image inference.
'''

SAVE_MD = '''## {number}. Save the feature dataset

The 2x2 spatial vectors and pooled vectors are saved in float32, bit-identical
to what the training notebook computes. The full feature map is saved in
float16 to halve its size (about 745 MB instead of 1.5 GB). The reported
relative conversion error shows the precision cost.
'''

SAVE_CODE = '''FEATURE_DIR = Path('/kaggle/working/trashnet_dinov3_convnext_tiny_features')
LOG_DIR = Path('/kaggle/working/run_logs')
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

file_summary, conversion_error = {}, {}
for name, features in raw_features.items():
    spatial = spatial_grid_vectors(features['feature_map'])
    maps16 = features['feature_map'].astype(np.float16)
    if not np.isfinite(maps16).all():
        raise FloatingPointError(f'{name}: float16 overflow in feature map')
    conversion_error[name] = float(np.max(np.abs(maps16.astype(np.float32) - features['feature_map']))
                                   / np.max(np.abs(features['feature_map'])))
    arrays = {f'{name}_spatial_2x2.npy': spatial, f'{name}_pooled.npy': features['pooled'],
              f'{name}_feature_map_f16.npy': maps16, f'{name}_labels.npy': features['labels']}
    for filename, array in arrays.items():
        np.save(FEATURE_DIR / filename, array)
        file_summary[filename] = {'dtype': str(array.dtype), 'shape': list(array.shape)}
    del maps16, spatial
print('float16 map relative max conversion error:', conversion_error)

manifest.to_csv(FEATURE_DIR / 'metadata.csv', index=False)
dataset_manifest = {
    'source_dataset': f'kaggle:{DATASET_HANDLE}/{DATASET_SUBDIR}',
    'backbone': MODEL_HANDLE, 'backbone_file': MODEL_FILENAME,
    'class_names': class_names,
    'split': {'method': 'baselines.split_dataset (HF train_test_split, stratify_by_column=label)',
              'seed': SEED, 'fractions': {'train': 0.70, 'validation': 0.15, 'test': 0.15},
              'sizes': {name: len(splits[name]) for name in ('train', 'validation', 'test')}},
    'preprocessing': 'native resolution (no resize/crop/pad); RGB/255; ImageNet mean/std',
    'inference': {'precision': 'float32', 'trainable_weights': 0, 'training_flag': False,
                  'replicas': strategy.num_replicas_in_sync,
                  'per_replica_batch': PER_REPLICA_EXTRACTION_BATCH,
                  'parity_relative_max_error': parity_error},
    'spatial_2x2': "tf.image.resize(feature_map, (2, 2), method='area') flattened row-major (2, 2, 768)",
    'feature_map_float16_relative_max_error': conversion_error,
    'files': file_summary,
}
(FEATURE_DIR / 'manifest.json').write_text(json.dumps(dataset_manifest, indent=2))
'''

README_CODE = '''readme = """# TrashNet: frozen DINOv3 ConvNeXt-Tiny features

Backbone outputs for every image of TrashNet (`feyzazkefe/trashnet`,
`dataset-resized`, 2,527 images, 512x384) from the frozen DINOv3 ConvNeXt-Tiny
Keras model `mataeoanderson/dinov3-convnext/keras/tiny/1`. They were computed at native
resolution with ImageNet normalization, float32 inference, and no augmentation.

Splits: seed-42 stratified 70/15/15 (train 1,768 / validation 379 / test 380),
identical to the OU DL1 Unit 3 and Unit 4 TrashNet notebooks.

Files per split (row i of each file is row i of that split in metadata.csv):
- `{split}_spatial_2x2.npy`: float32 (N, 3072). The 2x2 area-pooled feature map, row-major (2, 2, 768).
- `{split}_pooled.npy`: float32 (N, 768). The global pooled vector.
- `{split}_feature_map_f16.npy`: float16 (N, 12, 16, 768). The full last-stage feature map.
- `{split}_labels.npy`: int32 (N,). 0 cardboard, 1 glass, 2 metal, 3 paper, 4 plastic, 5 trash.

Load in a Kaggle notebook (attach this dataset first):

    import kagglehub, numpy as np
    from pathlib import Path
    root = Path(kagglehub.dataset_download('mataeoanderson/trashnet-dinov3-convnext-tiny-features'))
    root = next(p.parent for p in root.rglob('manifest.json'))
    x_train = np.load(root / 'train_spatial_2x2.npy'); y_train = np.load(root / 'train_labels.npy')
    maps = np.load(root / 'train_feature_map_f16.npy', mmap_mode='r')  # memory-mapped

All modeling in the source project uses TensorFlow / Keras.
"""
(FEATURE_DIR / 'README.md').write_text(readme)
'''

VERIFY_CODE = '''# Reload every saved array and confirm shapes, dtypes, labels, and row alignment.
for name in ('train', 'validation', 'test'):
    rows = manifest[manifest['split'] == name]
    labels = np.load(FEATURE_DIR / f'{name}_labels.npy')
    assert labels.tolist() == rows['label'].tolist()
    spatial = np.load(FEATURE_DIR / f'{name}_spatial_2x2.npy')
    pooled = np.load(FEATURE_DIR / f'{name}_pooled.npy')
    maps = np.load(FEATURE_DIR / f'{name}_feature_map_f16.npy', mmap_mode='r')
    assert spatial.shape == (len(rows), 3072) and spatial.dtype == np.float32
    assert pooled.shape == (len(rows), 768) and pooled.dtype == np.float32
    assert maps.shape[0] == len(rows) and maps.dtype == np.float16
    print(name, 'OK', spatial.shape, pooled.shape, maps.shape)

total_mb = sum(p.stat().st_size for p in FEATURE_DIR.iterdir()) / 2**20
print(f'Dataset folder: {FEATURE_DIR} | {total_mb:.1f} MiB')
for path in sorted(FEATURE_DIR.iterdir()):
    print(f'  {path.name:40s} {path.stat().st_size / 2**20:9.1f} MiB')

elapsed_seconds = time.perf_counter() - notebook_started
(LOG_DIR / 'notebook_runtime.json').write_text(json.dumps(
    {'execution_seconds': elapsed_seconds, 'extraction_seconds': extraction_seconds}, indent=2))
print(f'Notebook execution time: {elapsed_seconds / 60.:.2f} minutes')
'''


def extraction_notebook():
    cells = [md(EXTRACT_TITLE), md(DEPENDENCY_MD), code(DEPENDENCY_CODE),
             md('## 2. Configuration and GPU check'), code(CONFIG_IMPORTS)]
    cells += embedded_helper_cells(3)
    cells += [md(DATASET_MD.format(number=4)), code(DATASET_CODE),
              md(SPLIT_MD.format(number=5)), code(SPLIT_CODE),
              md(EXTRACTION_DATASET_MD.format(number=6)), code(BACKBONE_CODE),
              md(SAVE_MD.format(number=7)), code(SAVE_CODE), code(README_CODE), code(VERIFY_CODE)]
    return cells


def write(cells, path):
    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}}
    nbformat.validate(notebook)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    print('Wrote', path.relative_to(ROOT), f'({len(cells)} cells)')


if __name__ == '__main__':
    write(training_notebook(), ROOT / 'training' / 'Unit4_Training.ipynb')
    write(extraction_notebook(), ROOT / 'feature_extraction' / 'Unit4_Feature_Extraction.ipynb')
