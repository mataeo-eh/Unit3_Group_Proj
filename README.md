# Unit 3 group project

[Open the Kaggle notebook](https://www.kaggle.com/code/mataeoanderson/unit3-trashnet-dinov3-tiny-starter)

`Notebook.ipynb` is a minimal starter. It loads the full hosted training split of
[TrashNet](https://huggingface.co/datasets/garythung/trashnet), displays one image
and its dataset label, loads only
[DINOv3 ConvNeXt Tiny](https://www.kaggle.com/models/mataeoanderson/dinov3-convnext/keras/tiny),
and checks one forward pass. Training, classification, and evaluation remain
group project work. The backbone returns features rather than class predictions.

## Run on Kaggle

Enable **Internet** and attach the model
`mataeoanderson/dinov3-convnext/keras/tiny/1`, then run all cells. CPU is enough
for the starter. The first cell installs pinned dependencies matching the saved
Keras model. The dataset's published Parquet conversion commit and the model
version are also pinned. All four training shards are loaded explicitly to
preserve the hosted image/label schema and avoid folder-based label inference.

To publish updates using an authenticated Kaggle CLI, run from this directory:

```sh
kaggle kernels push -p .
kaggle kernels status mataeoanderson/unit3-trashnet-dinov3-tiny-starter
```

The local, Git-ignored `kernel-metadata.json` configures the public notebook,
Internet access, and the Tiny model attachment. It is not committed because it
contains the publishing account identifier. Other group members can upload
`Notebook.ipynb` through Kaggle's editor and configure the settings above, or
create their own local metadata with `kaggle kernels init -p .`.
No credentials belong in this repository.

## Expected result

The notebook shows a labeled image and prints `SUCCESS` after checking pooled
features of shape `(1, 768)` and spatial features of shape `(1, 7, 7, 768)`.
It also saves `outputs/sample_image.png` and `outputs/smoke_test.json` in the
Kaggle working directory for downloading as run evidence.

Kaggle version 3 completed successfully on October 8, 2026: 5,054 images,
six class names, a displayed `cardboard` sample, and both feature-shape checks
passed with the Tiny model. Notebook outputs are available on Kaggle; the Git
copy remains a clean starter without embedded runtime logs or large images.

Local Jupyter can run the same notebook with Internet access. Data and model
files are downloaded into library caches; runtime outputs and model weights
are ignored by Git. Only this project directory is the Git repository root.

## References

- [Hugging Face dataset loading](https://huggingface.co/docs/datasets/en/loading)
- [KaggleHub model downloads](https://github.com/Kaggle/kagglehub#download-models)
- [Keras model loading](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/)
- [Kaggle notebook metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)

The model bundle includes Meta's DINOv3 license; refer to the model card and
bundled license for its terms.
