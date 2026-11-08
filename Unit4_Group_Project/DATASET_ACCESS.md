# Dataset access and notebook inputs

Use **TrashNet DINOv3 ConvNeXt-Tiny Features** for experiments that train
classifier heads on precomputed features. Version **1** was confirmed public
on **November 8, 2026**: Kaggle returned its dataset metadata without any
credentials and reported `isPrivate: false`. The file listing also contains
the feature arrays, labels, `metadata.csv`, `manifest.json`, and `README.md`.

## Attach it in the Kaggle editor

1. Sign in to your own Kaggle account and open the notebook you want to train
   in edit mode. These steps apply to a new notebook, a copy, or an existing
   notebook that you are revising.
2. Open the **Input** pane and select **Add Input**. Search for
   **TrashNet DINOv3 ConvNeXt-Tiny Features**. If several results have similar
   titles, use the exact dataset link supplied privately by your teammate.
3. Add that dataset and confirm it appears in the notebook's Input pane.
   For a new notebook, an alternative is to open the dataset page, select
   **Code**, then **Create a notebook**.
4. Expand the attached dataset and confirm it contains `manifest.json`,
   `metadata.csv`, `train_spatial_2x2.npy`, `train_labels.npy`, and the
   corresponding `validation_*` and `test_*` files. Use the actual mounted
   file paths shown by Kaggle when configuring your loader.
5. Record the attached dataset version in your experiment notes. Version 1
   is the verified baseline. Check the inputs again before **Save Version**
   → **Save & Run All**, especially after copying or importing a notebook.
   A saved notebook version includes a snapshot of its data sources;
   changes to inputs must be saved in a new notebook version.

A public dataset can be attached from a partner's own account without
sharing the uploader's credentials. If access fails, open the privately
supplied link, confirm that it is the same dataset, and recheck its current
visibility. The verification above records a point in time.

## When uploading notebooks through the Kaggle CLI

For each notebook you publish, include the dataset's exact `owner/dataset-slug`
in the `dataset_sources` list of that notebook's local
`kernel-metadata.json`. Obtain the handle from the privately supplied link.
Add it alongside any other inputs that notebook still uses:

```json
{
  "dataset_sources": ["<owner>/<dataset-slug>"]
}
```

This is a field example, not a complete metadata file. Keep the notebook's
own ID, code file, settings, and other source lists. Use your own account
for the notebook ID. The project already Git-ignores `kernel-metadata.json`.

## Feature inputs for training

The existing `training/Unit4_Training.ipynb` loads images and runs the frozen
backbone. Adding this dataset does not make it consume cached arrays
automatically. A notebook designed for cached features should load matching
feature and label arrays and bypass image loading and backbone extraction.
Keep TensorFlow / `tensorflow.keras` for modeling.

| File pattern | Shape / dtype | Purpose |
|---|---|---|
| `{split}_spatial_2x2.npy` | `(N, 3072)`, float32 | Existing Unit 3/4 classifier input |
| `{split}_labels.npy` | `(N,)`, int32 | Corresponding class labels |
| `{split}_pooled.npy` | `(N, 768)`, float32 | Alternative global pooled representation |
| `{split}_feature_map_f16.npy` | `(N, 12, 16, 768)`, float16 | Full maps for other spatial poolings |

Splits are `train` (1,768 rows), `validation` (379), and `test` (380), with
the original seed-42 stratified 70/15/15 assignment. Preserve row alignment
with `metadata.csv` and the class order in `manifest.json`: cardboard,
glass, metal, paper, plastic, trash (labels 0–5). Use train and validation
for model development; reserve test data for the final evaluation.

## Public documentation and account privacy

This guide omits uploader handles, personal names, and account-specific
links. Supply the exact link and handle privately. Do not publish local
credentials, account metadata, or the private handoff note.

Kaggle still displays the dataset owner's profile, and a public notebook's
attached inputs can lead readers to that profile. Omitting the link here
does not make the dataset or its use anonymous. Anyone who follows the
dataset title can also find its owner. Existing published artifacts and
repository history may already contain account references.

## Platform references

- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks):
  adding data sources and saved notebook versions.
- [Kaggle CLI notebook metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md):
  the `dataset_sources` field and notebook publishing metadata.
