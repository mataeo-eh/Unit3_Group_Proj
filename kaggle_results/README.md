# Kaggle execution archive

`Notebook_finished.ipynb` is the output-inclusive notebook downloaded from
Kaggle version 4 on October 11, 2026. The execution ended with ERROR, rather than
successful training: a native-resolution image exhausted GPU memory during
frozen-backbone feature extraction (`ResourceExhaustedError` in `Dense.call`).
Neither prediction head was trained, and no final metrics were produced.
The notebook preserves the executed cells and original error output for review.
The `_finished` suffix records that this execution has ended.

Only the notebook was downloaded; model artifacts were not downloaded.
