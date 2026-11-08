"""Embed reusable baseline helpers into the standalone Kaggle notebook.

Run after editing baselines.py. Preserve dependency, dataset, and image EDA cells;
replace only the previous smoke-test and placeholders with the experiment flow.
"""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
notebook = nbformat.read(ROOT / 'Notebook.ipynb', as_version=4)
cells = notebook.cells[:10]
cells[1].source = '''## 1. Compatible dependencies and configuration

Keep TensorFlow 2.20, required by Kaggle's existing TensorFlow add-ons, while
using Keras 3.15.1 to read the saved artifact. Pin Protobuf 5.29.5 and Hugging
Face Hub 1.31.0 within all reported consumers' supported ranges. Install the
missing BigQuery Storage dependency and compatible Gym version. Run `pip check`
before imports: dependency conflicts fail setup instead of being ignored.
Remove unused Dopamine (whose Gym requirement conflicts with Gymnax) and Google
Colab (whose pandas/Jupyter requirements conflict with Kaggle's server stack).
Remove unused legacy MoviePy, whose Decorator requirement conflicts with MNE.
'''
cells[2].source = '''# Install compatible pins before importing TensorFlow or dataset libraries.
import subprocess
import sys
import time

# Include dependency setup, extraction, training, scoring, and plots in runtime.
# perf_counter measures elapsed execution time; Kaggle queue time is excluded.
notebook_started = time.perf_counter()

# These unused preinstalled distributions impose mutually incompatible runtime
# constraints; removing them avoids changing Kaggle's pandas/Jupyter server.
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y',
                'dopamine-rl', 'google-colab', 'moviepy'], check=True)
requirements = [
    'tensorflow==2.20.0', 'keras==3.15.1', 'datasets==5.1.0',
    'kagglehub==1.0.2', 'Pillow==12.3.0', 'protobuf==5.29.5',
    'huggingface-hub==1.31.0', 'google-cloud-bigquery-storage==2.35.0',
    'gym==0.26.2', 'decorator==5.3.1',
]
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                '--disable-pip-version-check', *requirements], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
'''
cells[0].source = '''# Unit 3 group project: TrashNet + DINOv3 ConvNeXt Tiny

Compare frozen native-resolution global pooled features with a linear softmax
classifier, an 8-unit GELU ANN, a larger ANN with dropout, and a training-majority
baseline. Test images are
held out for the group's eventual single final model. All modeling uses TensorFlow/Keras.
'''
cells[3].source = cells[3].source.replace('IMAGE_SIZE = (224, 224)\n', '')
cells[3].source += '\nSEED = 42\nBATCH_SIZE = 64\nMAX_EPOCHS = 50\nPATIENCE = 5\nkeras.utils.set_random_seed(SEED)\n'
# Avoid duplicating configuration when regenerating an already built notebook.
config = cells[3].source.split('\nSEED = 42')[0]
if "set_memory_growth" not in config:
    config = config.replace('import tensorflow as tf', '''import tensorflow as tf
# Configure allocation before loading weights or creating any GPU tensors.
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)''')
cells[3].source = config + '\nSEED = 42\nBATCH_SIZE = 64\nMAX_EPOCHS = 50\nPATIENCE = 5\nkeras.utils.set_random_seed(SEED)\n'
cells += [nbformat.v4.new_markdown_cell('''## 5. Reusable experiment helpers

These helpers are maintained in `baselines.py` and embedded here by
`build_notebook.py` so this notebook runs by itself on Kaggle. `build_head` accepts
any fixed feature width; a spatial-feature experiment can reuse the head after
choosing its spatial representation/aggregation. No resize, crop, or padding is
applied. RGB scaling and ImageNet mean/std normalization remain.

API references: [Keras callbacks](https://keras.io/api/callbacks/),
[batch learning-rate schedules](https://keras.io/api/optimizers/learning_rate_schedules/learning_rate_schedule/),
[ROC/PR AUC](https://keras.io/api/metrics/classification_metrics/),
[Dataset stratified splitting](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset.train_test_split).
'''), nbformat.v4.new_code_cell((ROOT / 'baselines.py').read_text()),
nbformat.v4.new_markdown_cell('''## 6. Stratified splits and class counts

The pinned Parquet conversion contains only `train`. Create 70% train / 15%
validation / 15% test with seed 42 and stratified labels. The reusable splitter
also accepts author-provided train/test or train/validation/test partitions and
preserves them, splitting train only when a partition is missing. Save row IDs
so future experiments use exactly the same partitions. Only counts/IDs are
inspected for test; its images never reach the backbone or classifiers.
'''), nbformat.v4.new_code_cell('''import pandas as pd

splits = split_dataset(DatasetDict({'train': dataset}), seed=SEED)
counts = pd.DataFrame({name: np.bincount(np.asarray(data['label']), minlength=len(class_names))
                       for name, data in splits.items()}, index=class_names)
display(counts)
print('Split sizes:', {name: len(data) for name, data in splits.items()})
(OUTPUT_DIR / 'split_manifest.json').write_text(json.dumps(
    {'seed': SEED, 'dataset_revision': DATASET_REVISION,
     'row_ids': {name: list(data['row_id']) for name, data in splits.items()}}, indent=2))
'''), nbformat.v4.new_markdown_cell('''## 7. Extract native-resolution pooled features once

One forward pass per train/validation image, performed before any head trains.
All three heads share the same float32 RAM cache. Spatial maps are discarded. The
backbone is fully frozen and called with `training=False`. No test feature cache
is created, and the former extra sample forward pass has been removed.

Use a pooled-only TensorFlow graph to avoid retaining eager intermediate maps.
GPU convolution/dense activations use mixed float16; pretrained variables,
normalization, global pooling, cached vectors, and classifier heads stay float32.
This reduces memory without resizing or changing any pretrained weight.
'''), nbformat.v4.new_code_cell('''import gc

use_mixed_precision = bool(tf.config.list_physical_devices('GPU'))
backbone = prepare_pooled_backbone(backbone, mixed_precision=use_mixed_precision)
gc.collect()
print('Backbone execution: pooled-only graph | Mixed activations:', use_mixed_precision)
print('Backbone input:', backbone.input_shape, '| Output:', backbone.output_shape)
feature_cache, extraction_report = cache_pooled_features(backbone, splits)
extraction_report['mixed_precision_activations'] = use_mixed_precision
(OUTPUT_DIR / 'feature_extraction.json').write_text(json.dumps(extraction_report, indent=2))
'''), nbformat.v4.new_markdown_cell('''## 8. Train linear softmax, then small and larger ANNs

All three use batch size 64, at most 50 epochs, and strict validation-loss improvement
with patience 5. Best checkpoints are saved at each improving epoch and reloaded
for final scoring. All epoch losses, including patience epochs, remain available.

The linear head is multinomial logistic regression (768 → 6 softmax), with 4,614
parameters. The small ANN is 768 → 8 GELU → 6 softmax, with 6,206 parameters.
The larger ANN is 768 → 40 GELU → dropout(0.1) → 24 GELU → dropout(0.1) → 6
softmax, with 31,894 parameters. This tests increased capacity with regularization;
capacity and dropout change together, so this is not a capacity-only ablation.
Adam uses 0.001 for all; both ANNs use linear WSD across planned optimizer updates:
10% warmup, 70% stable, 20% decay to 1% of peak. Early stopping can end before decay.
Add future configurations to `RUN_SPECS` to reuse training, evaluation, and plots.
'''), nbformat.v4.new_code_cell('''RUN_SPECS = [
    {'name': 'logistic_regression', 'hidden_units': (), 'learning_rate': 0.001, 'wsd': False},
    {'name': 'tiny_ann', 'hidden_units': (8,), 'learning_rate': 0.001, 'wsd': True},
    {'name': 'larger_ann', 'hidden_units': (40, 24), 'dropout_rates': (0.1, 0.1),
     'learning_rate': 0.001, 'wsd': True},
]
trained_heads, histories = {}, {}
for spec in RUN_SPECS:
    print('Training:', spec['name'], flush=True)
    trained_heads[spec['name']], histories[spec['name']] = train_head(
        spec, feature_cache, len(class_names), OUTPUT_DIR,
        batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED)
    print({key: value for key, value in histories[spec['name']].items() if key != 'history'})
'''), nbformat.v4.new_markdown_cell('''## 9. Final train/validation metrics and comparison

Final scores use reloaded best-validation checkpoints. Predictions are softmax
argmax. Precision, recall, F1, ROC-AUC, and PR-AUC are macro averages; per-class
scores and confusion matrices are also retained. AUC uses Keras interpolation at
2,000 thresholds for each one-vs-rest class. These are numerical approximations.

The majority class is selected from training labels only and always receives
probability 1. Keras clips zero probabilities for finite cross-entropy reporting;
ideal wrong-class loss for this hard baseline would be infinite. No test scoring
occurs. Training metrics are diagnostics; validation is the comparison set.
'''), nbformat.v4.new_code_cell('''majority_class = int(np.bincount(feature_cache['train'][1], minlength=len(class_names)).argmax())
print('Training majority class:', class_names[majority_class])
results = {'majority_class': {}}
for split_name, (x, y) in feature_cache.items():
    probabilities = np.zeros((len(y), len(class_names)), dtype=np.float32)
    probabilities[:, majority_class] = 1.
    results['majority_class'][split_name] = evaluate_probabilities(y, probabilities, class_names)
for name, head in trained_heads.items():
    results[name] = {}
    for split_name, (x, y) in feature_cache.items():
        probabilities = head.predict(feature_dataset(x, y, BATCH_SIZE), verbose=0)
        results[name][split_name] = evaluate_probabilities(y, probabilities, class_names)

metric_names = ['accuracy', 'f1_macro', 'pr_auc_macro', 'roc_auc_macro', 'recall_macro', 'precision_macro', 'loss']
comparison = pd.DataFrame([{'model': name, 'split': split_name,
                           **{key: report[key] for key in metric_names}}
                          for name, reports in results.items() for split_name, report in reports.items()])
display(comparison.set_index(['split', 'model']).sort_index())
comparison.to_csv(OUTPUT_DIR / 'comparison.csv', index=False)
(OUTPUT_DIR / 'metrics.json').write_text(json.dumps(results, indent=2))
for name, reports in results.items():
    print(name, '— validation per-class metrics')
    display(pd.DataFrame(reports['validation']['per_class']).set_index('class'))
'''), nbformat.v4.new_markdown_cell('''## 10. Learning curves, confusion matrices, ROC and PR curves

Every run gets the same plots. Learning curves retain all completed epochs and
mark the best checkpoint. Overfitting may appear as training loss falling while
validation loss rises; this behavior is not guaranteed in five patience epochs.
'''), nbformat.v4.new_code_cell('''plot_reports(results, histories, class_names, OUTPUT_DIR)
print('Saved artifacts:', [path.name for path in sorted(OUTPUT_DIR.iterdir())])
print('Finished. Test inference and scoring calls: 0.')
# Save and display full execution time so the downloaded notebook carries it.
elapsed_seconds = time.perf_counter() - notebook_started
runtime_report = {'execution_seconds': elapsed_seconds,
                  'execution_minutes': elapsed_seconds / 60., 'queue_time_included': False}
(OUTPUT_DIR / 'notebook_runtime.json').write_text(json.dumps(runtime_report, indent=2))
print(f'Notebook execution time: {elapsed_seconds / 60.:.2f} minutes ({elapsed_seconds:.1f} seconds).')
''')]
notebook.cells = cells
nbformat.validate(notebook)
nbformat.write(notebook, ROOT / 'Notebook.ipynb')
