# Unit 4 group project: TrashNet class imbalance

Unit 4 starts from what worked best in Unit 3. It keeps the frozen DINOv3
ConvNeXt-Tiny backbone and the **2×2 spatial feature map** (3,072 values, which
beat the 768-value pooled vector). It also uses the **larger prediction head**
(40 GELU → dropout 0.1 → 24 GELU → dropout 0.1 → 6). The goal is the class
imbalance that limited macro F1: trash has 95 of 1,768 training images.

All modeling is TensorFlow / Keras 3. No other ML framework is used.

## Layout

| Path | Purpose |
|---|---|
| `unit4_helpers.py` | Two-GPU extraction, logit heads, focal / logit-adjusted losses, training, reporting |
| `../baselines.py` | Unit 3 helpers reused unchanged: split, preprocessing, `WarmStableDecay`, `evaluate_probabilities` |
| `build_notebooks.py` | Generates both notebooks, embedding `../baselines.py` and `unit4_helpers.py` |
| `training/` | `Unit4_Training.ipynb` + its Kaggle `kernel-metadata.json` |
| `feature_extraction/` | `Unit4_Feature_Extraction.ipynb` + its Kaggle `kernel-metadata.json` |

After editing a helper, run `python build_notebooks.py` to regenerate both notebooks.

## Training notebook (`training/`)

| Run | Loss | Optimizer |
|---|---|---|
| `control_ce` | cross entropy (Unit 3 loss) | Adam |
| `exp1_focal` | alpha-balanced focal, γ = 2, α = N / (K·n_c) | Adam |
| `exp2_logit_adjusted` | logit-adjusted CE, τ = 1, label smoothing 0.1 | AdamW, weight decay 0.05 (no bias decay) |

Every run uses the same settings:

- Global batch 256, split as 128 per T4 through `MirroredStrategy`.
- WSD learning rate: 10% warmup, 70% stable, 20% decay to 1% of a 2e-3 peak.
- 100 epochs, giving 700 optimizer steps.

Each run saves `<name>_best.keras` (best validation loss) and `<name>_last.keras`
(end of the WSD decay). Both are scored on train, validation and test.

Design notes:

- **Batch size without gradient accumulation.** The heads train on cached
  feature vectors. A 256-row batch is about 3 MB and the head has about 124k
  parameters, so memory is not a limit. Peak LR is scaled by √4 (1e-3 → 2e-3).
- **No early stopping.** In Unit 3, early stopping always fired before the
  WSD decay phase began. Training the full schedule makes the last-epoch head
  the annealed WSD endpoint, which is a meaningful comparison with best-val.
- **Experiment 2 rationale.** The frozen backbone makes this a decoupled
  classifier-learning problem (Kang et al., ICLR 2020). Logit adjustment
  (Menon et al., ICLR 2021), which is the same as Balanced Softmax at τ = 1, is
  Fisher-consistent for balanced error. That matches the "equal performance
  across classes" goal. Label smoothing (MiSLAS, CVPR 2021) and classifier
  weight decay (weight balancing, CVPR 2022) address the overconfidence and
  per-class weight-norm bias that imbalance causes. Resampling is left out
  because combined with logit adjustment it over-corrects.
- `control_ce` isolates the effect of the loss from the batch, schedule and
  head changes.

## Feature dataset notebook (`feature_extraction/`)

This notebook runs the backbone once on all 2,527 images, using both GPUs, and
writes `trashnet_dinov3_convnext_tiny_features/`:

- `{split}_spatial_2x2.npy`: float32, (N, 3072). The Unit 3/4 head input.
- `{split}_pooled.npy`: float32, (N, 768).
- `{split}_feature_map_f16.npy`: float16, (N, 12, 16, 768). The full map, for other spatial poolings.
- `{split}_labels.npy`, plus `metadata.csv`, `manifest.json` and `README.md`.

The split is the same seed-42 stratified 70/15/15 split, and row order matches
`metadata.csv`. The folder is published as the public Kaggle dataset
**TrashNet DINOv3 ConvNeXt-Tiny Features**. Public access to version 1 was
verified without credentials on November 8, 2026.

See [Dataset access and notebook inputs](DATASET_ACCESS.md) for instructions
for attaching it to new, copied, or revised notebooks. Later notebooks can
load the saved arrays with `np.load` and skip the backbone entirely. The
current `training/Unit4_Training.ipynb` still extracts features from images;
attaching the feature dataset alone does not switch that notebook to cached
features. The dataset's README has a loading snippet.

To refresh the dataset after re-running the notebook:

```sh
kaggle kernels output <owner>/<extraction-notebook-slug> -p <dir>
kaggle datasets version -p <dir>/trashnet_dinov3_convnext_tiny_features -m "<message>"
```

## Kaggle

```sh
kaggle kernels push -p training --accelerator NvidiaTeslaT4            # GPU T4 x2
kaggle kernels push -p feature_extraction --accelerator NvidiaTeslaT4
```

Kaggle's November 2026 GPU image ships TF 2.21 and protobuf 7.36. Unit 3's
TF 2.20 / protobuf 5.29 pins now fail `pip check`. The notebooks keep the
shipped TF and change only `keras==3.15.1`, `datasets==5.1.0`,
`Pillow==12.3.0`, `google-cloud-bigquery-storage==2.43.0` and `ydf==0.17.0`.
They also uninstall several unused packages: Dopamine, Colab, MoviePy and
RAPIDS cudf/dask-cudf/cuml. The notebook markdown explains each change.

Both kernels need Internet (for the dependency pins) and attach two inputs:
the original TrashNet image dataset and the frozen DINOv3 ConvNeXt-Tiny
Keras model. Use the local publishing metadata for their exact handles.
The notebooks raise an error unless exactly two GPUs are visible.
`kernel-metadata.json` is Git-ignored because it holds the account id.
Download finished runs with `../download_finished_notebook.py`, which checks
COMPLETE status and that no cell errored.

## References

- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017 — <https://arxiv.org/abs/1708.02002>
- Kang et al., *Decoupling Representation and Classifier for Long-Tailed Recognition*, ICLR 2020 — <https://arxiv.org/abs/1910.09217>
- Ren et al., *Balanced Meta-Softmax for Long-Tailed Visual Recognition*, NeurIPS 2020 — <https://arxiv.org/abs/2007.10740>
- Menon et al., *Long-Tail Learning via Logit Adjustment*, ICLR 2021 — <https://arxiv.org/abs/2007.07314>
- Zhong et al., *Improving Calibration for Long-Tailed Recognition* (MiSLAS), CVPR 2021 — <https://arxiv.org/abs/2104.00466>
- Alshammari et al., *Long-Tailed Recognition via Weight Balancing*, CVPR 2022 — <https://arxiv.org/abs/2203.14197>
- Zhu et al., *Generalized Logit Adjustment*, NeurIPS 2023 — <https://arxiv.org/abs/2310.08106>
- Terven et al., *A Strong Balanced-Softmax Classifier-Retraining Baseline*, 2026 — <https://arxiv.org/abs/2607.09832>
- Keras `CategoricalFocalCrossentropy` — <https://keras.io/api/losses/probabilistic_losses/#categoricalfocalcrossentropy-class>
- TensorFlow distributed training with Keras — <https://www.tensorflow.org/tutorials/distribute/keras>
