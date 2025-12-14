# Relating Loss Function Scaling in Neural Language Models to Hilberg's Conjecture

## IMS International Conference on Statistics and Data Science (ICSDS 2025)
[![ICSDS 2025](https://img.shields.io/badge/Conference-ICSDS%202025-blue)](https://sites.google.com/view/ims-icsds2025/)
[![Cite this](https://img.shields.io/badge/Cite-CFF-green)](./CITATION.cff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Dataset and Models](https://img.shields.io/badge/Dataset%20%26%20Models-Hugging%20Face-orange)](https://huggingface.co/collections/AdrianRasoOnHF/hilberg-scaling-in-neural-language-models-icsds-2025)


This repo contains the materials related to my talk at IMS ICSDS (2025), including everything from the [slides](talk/presentation.pdf) of the presentation to the [code](src/) that implements the methodology’s pipeline. This way, it allows readers to explore the contents of the talk and reproduce the full experiment.

## Abstract

> Hilberg's conjecture posits that block entropy grows sublinearly in natural language data, suggesting a vanishing asymptotic entropy rate and unbounded excess entropy in the limit. This behavior is puzzling and has motivated numerous studies in the literature concerning the theoretical and practical consequences of this growth, having been suggested as an issue for machine learning models. In this work, we ask how this growth interacts with training in neural language models. Using the July 20, 2025 English Wikipedia dump, we estimate block entropies H(n) using per-token code lengths from a bias-corrected Prediction by Partial Matching (PPM) estimator, reporting 95\% confidence intervals from variance estimation across 50 disjoint text subsets. We train autoregressive GPT neural models with varied capacity and context length to measure the conditional entropy loss and study how its decrease may relate to the discrete derivative of H(n) and the Hilberg β exponent. The results show that the empirical baseline exhibits subextensive behavior consistent with Hilberg's scaling, and that Transformer LMs display a robust power law improvement of conditional entropy with increasing context, with fitted exponents in a similar subextensive regime.

## Datasets and Models

All datasets and trained models produced for this study are publicly released to ensure transparency and reproducibility. These are stored under a Hugging Face collection [here](https://huggingface.co/collections/AdrianRasoOnHF/hilberg-scaling-in-neural-language-models-icsds-2025).

### Dataset

- [`wikidump-en-2025-07`](https://huggingface.co/datasets/AdrianRasoOnHF/wikidump-en-2025-07) (7,044,210 articles)

### Models
- [`gpt-hilberg-1M`](https://huggingface.co/AdrianRasoOnHF/gpt-hilberg-1M) (1M params, 128 context length)
- [`gpt-hilberg-5M`](https://huggingface.co/AdrianRasoOnHF/gpt-hilberg-5M) (5M params, 512 context length)
- [`gpt-hilberg-10M`](https://huggingface.co/AdrianRasoOnHF/gpt-hilberg-10M) (10M params, 1024 context length)

## Code

For details on the method that these scripts codify, you can check [the talk slides](talk/presentation.pdf).

### Data
- [`download.py`](src/data/download.py)
- [`extract.py`](src/data/extract.py)
- [`clean.py`](src/data/clean.py)
- [`build_dataset.py`](src/data/build_dataset.py)

### Train
- [`gpt_model.py`](src/train/gpt_model.py)
- [`train_all_models.py`](src/train/train_all_models.py)

### Analysis
- [`PPM-D_estimator.py`](src/analysis/PPM-D_estimator.py)
- [`lm_hilberg_scaling.py`](src/analysis/lm_hilberg_scaling.py)
- [`pre-asymptotic_true_v_models.py`](src/analysis/pre-asymptotic_true_v_models.py)

## Talk resources
- `talk/presentation.pdf` (`.tex` included)
- `talk/abstract/abstract.pdf` (`.tex` included)

## Reproduction
Clone the repository and install the project in editable mode with
```bash
pip install -e .
```
To perform the dataset construction, downloading, extracting, and preprocessing the July 20, 2025 English Wikipedia dump, perform:
```bash
python src/data/download.py
python src/data/extract.py
python src/data/clean.py
```
or the wrapper
```bash
python src/data/build_dataset.py
```
This task can take several days to complete, so it is advisable to obtain directly the dataset from my Hugging Face profile (see [Dataset](#dataset) section above!).

To run the entropy estimation, you can run:
```bash
python src/analysis/PPM-D_estimator.py
```

To train the models, the following line will do:
```bash
python src/train/train_all_models.py
``` 

To study the entropy scaling of the models, you should:
```bash
python src/analysis/lm_hilberg_scaling.py
```

To perform a pre-asymptotic analysis of these values and compare them with the true entropy scaling of the data, use:

```bash
python src/analysis/pre-asymptotic_true_v_models.py
```

These scripts, in the order presented here, constitute the complete method pipeline used in the experiment of the talk. You can see its exposition in [the presentation](talk/presentation.pdf).

## Citation
If you use this code, dataset, or analysis in your work, please cite the corresponding IMS ICSDS 2025 talk. See [`CITATION.cff`](./CITATION.cff) for more details on that.

## License

You can find the license of the project [here](./LICENSE).
