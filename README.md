# Unit 3 group project

`Notebook.ipynb` preserves the TrashNet loading, class display, and example image,
then trains two TensorFlow/Keras baselines on frozen DINOv3 ConvNeXt Tiny pooled
features. Modeling and metrics use TensorFlow/Keras; no PyTorch is used.

Images keep their native resolution. Preprocessing converts RGB pixels to 0–1
and applies ImageNet mean/std normalization, without resizing, cropping, or
padding. The exact saved backbone has input `(None, None, None, 3)` and returns
`pooled` `(N, 768)` and `feature_map` `(N, H', W', 768)`.

## Experiment design

- The pinned published Parquet conversion has one training split. Seeded,
  stratified splits allocate 70% train, 15% validation, and 15% held-out test.
  `split_dataset` also honors existing author splits when supplied a DatasetDict.
- Extract each train/validation image once before training. Both heads share
  float32 pooled vectors in RAM. The fully frozen backbone uses inference mode.
  Test images are never inferred, cached, or scored.
- Linear multinomial logistic regression: Dense(6, softmax), 4,614 parameters.
- Small ANN: Dense(8, GELU) → Dense(6, softmax), 6,206 parameters.
- Train heads sequentially with batch size 64, Adam, at most 50 epochs, and
  validation-loss early stopping with patience 5. Save every strictly improving
  checkpoint and reload the best checkpoint for final reports.
- ANN learning rate: peak 0.001, 10% linear warmup, 70% stable, 20% linear decay
  to 1% of peak across planned batches. Early stopping may precede decay.
- The majority baseline always predicts the training-majority class. Its hard
  one-hot probabilities use Keras epsilon clipping for finite reported loss.

Train and validation reports include accuracy, loss, macro precision/recall/F1,
macro one-vs-rest ROC-AUC and PR-AUC, per-class metrics, confusion matrices, and
ROC/PR plots. AUC uses Keras interpolation with 2,000 thresholds. All completed
epoch losses remain in the histories and learning curves, including patience
epochs. Validation is used for model comparison; training scores are diagnostic.
The test set is reserved for the group's eventual single selected model.

## Reuse and verification

`baselines.py` contains the reusable head factory, feature caching, splitter,
training, and reporting. `build_head` accepts any fixed-width feature vector;
spatial experiments should choose their own representation/aggregation before
calling the same factory. Add configurations to `RUN_SPECS` for more comparisons.
The notebook embeds the helpers so it runs independently on Kaggle.

After editing helpers, regenerate the notebook:

```sh
python build_notebook.py
python -m unittest test_baselines.py
```

Regression checks cover native pixels, cache call counts/test exclusion,
stratification, honoring author splits, WSD boundaries/serialization, parameter
counts, checkpoint reloads, and majority metrics.

## Run on Kaggle

Enable Internet and attach `mataeoanderson/dinov3-convnext/keras/tiny/1`. The
notebook pins TensorFlow/Keras versions matching the model serialization, the
published dataset conversion revision, and model version. A GPU accelerates the
single native-resolution feature-extraction pass.

With an authenticated Kaggle CLI and local `kernel-metadata.json`:

```sh
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels status mataeoanderson/unit3-trashnet-dinov3-tiny-starter
```

Publishing metadata is Git-ignored because it contains the publishing account
identifier. Partners can upload the notebook through Kaggle's editor or create
their own metadata with `kaggle kernels init -p .`. Credentials do not belong here.

## Outputs

Kaggle keeps `outputs/logistic_regression_best.keras` and
`outputs/tiny_ann_best.keras`, epoch CSV/JSON histories, split row IDs, extraction
counts, comparison CSV, metrics JSON, and plots. Trained heads remain on Kaggle;
the requested local deliverable is the executed notebook, named
`kaggle_results/Notebook_finished.ipynb` when the run finishes during monitoring.
The ordinary CLI `kernels pull` retrieves source without executed cell outputs.

## References

- [TrashNet dataset](https://huggingface.co/datasets/garythung/trashnet)
- [Tiny Keras backbone](https://www.kaggle.com/models/mataeoanderson/dinov3-convnext/keras/tiny)
- [Keras callbacks](https://keras.io/api/callbacks/)
- [Keras learning-rate schedules](https://keras.io/api/optimizers/learning_rate_schedules/learning_rate_schedule/)
- [Keras classification metrics](https://keras.io/api/metrics/classification_metrics/)
- [Dataset stratified splitting](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset.train_test_split)
