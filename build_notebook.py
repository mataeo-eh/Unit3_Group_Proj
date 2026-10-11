"""Embed reusable baseline helpers into the standalone Kaggle notebook.

Run after editing baselines.py. Preserve dependency, dataset, and image EDA cells;
replace only the previous smoke-test and placeholders with the experiment flow.
"""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
notebook = nbformat.read(ROOT / 'Notebook.ipynb', as_version=4)
cells = notebook.cells[:10]
cells[0].source = '''# Unit 3 group project: TrashNet + DINOv3 ConvNeXt Tiny

Compare frozen native-resolution global pooled features with a linear softmax
classifier, an 8-unit GELU ANN, and a training-majority baseline. Test images are
held out for the group's eventual single final model. All modeling uses TensorFlow/Keras.
'''
cells[3].source = cells[3].source.replace('IMAGE_SIZE = (224, 224)\n', '')
cells[3].source += '\nSEED = 42\nBATCH_SIZE = 64\nMAX_EPOCHS = 50\nPATIENCE = 5\nkeras.utils.set_random_seed(SEED)\n'
# Avoid duplicating configuration when regenerating an already built notebook.
config = cells[3].source.split('\nSEED = 42')[0]
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

One forward pass per train/validation image, performed before either head trains.
Both heads share the same float32 RAM cache. Spatial maps are discarded. The
backbone is fully frozen and called with `training=False`. No test feature cache
is created, and the former extra sample forward pass has been removed.
'''), nbformat.v4.new_code_cell('''feature_cache, extraction_report = cache_pooled_features(backbone, splits)
(OUTPUT_DIR / 'feature_extraction.json').write_text(json.dumps(extraction_report, indent=2))
'''), nbformat.v4.new_markdown_cell('''## 8. Train linear softmax, then lightweight ANN

Both use batch size 64, at most 50 epochs, and strict validation-loss improvement
with patience 5. Best checkpoints are saved at each improving epoch and reloaded
for final scoring. All epoch losses, including patience epochs, remain available.

The linear head is multinomial logistic regression (768 → 6 softmax). The ANN is
768 → 8 GELU → 6 softmax: 6,206 parameters versus 4,614 for the linear head. Adam
uses 0.001 for both; the ANN uses linear WSD across planned optimizer updates:
10% warmup, 70% stable, 20% decay to 1% of peak. Early stopping can end before decay.
Add future configurations to `RUN_SPECS` to reuse training, evaluation, and plots.
'''), nbformat.v4.new_code_cell('''RUN_SPECS = [
    {'name': 'logistic_regression', 'hidden_units': (), 'learning_rate': 0.001, 'wsd': False},
    {'name': 'tiny_ann', 'hidden_units': (8,), 'learning_rate': 0.001, 'wsd': True},
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
''')]
notebook.cells = cells
nbformat.validate(notebook)
nbformat.write(notebook, ROOT / 'Notebook.ipynb')
