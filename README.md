# AI Rewriting and Authorship Verification with AdHominem

This repository contains the code, experiment scripts, and selected result files used in my master's thesis project on the impact of AI rewriting on authorship verification.

The project is based on the original **AdHominem** model, an attention-based Siamese neural network for authorship verification in social media texts. In this thesis project, the original AdHominem pipeline was adapted to study whether AI-rewritten texts affect model performance, prediction confidence, and linguistic features.

---

## Project Overview

Authorship verification is the task of determining whether two texts were written by the same author. Since this task often relies on writing style, AI rewriting may create a challenge: if AI tools smooth, normalize, or change writing style, it may become harder for authorship verification models to make reliable decisions.

This project investigates how AI-rewritten texts affect authorship verification using Reddit-style text pairs and the AdHominem model.

The main research question is:

> If AI rewriting changes writing style, can authorship verification models still reliably identify whether two texts were written by the same author?

---

## Thesis

This repository was created as part of my master's thesis project in Computer Science. The thesis focuses on the impact of AI rewriting on authorship verification, especially model robustness, prediction confidence, and linguistic changes.

The full thesis PDF is not included in this repository. Instead, this repository provides the experiment scripts, selected result files, and analysis outputs needed to understand the technical workflow.

---

## Main Modifications

This repository keeps the core AdHominem model files and adds thesis-specific scripts and result files.

The main additions include:

* Scripts for creating clean Reddit authorship verification splits.
* Scripts for generating AI-rewritten training and test data.
* Scripts for building different training variants.
* Scripts for exporting data into AdHominem-compatible format.
* Scripts for checking dataset and rewrite consistency.
* Scripts for linguistic and topic-level analysis.
* Selected prediction outputs and result figures.

Unused pilot scripts and temporary files were removed to keep the repository clean.

---

## Repository Structure

The original AdHominem model files are kept in the root directory.

The thesis-specific files are organized as follows:

* `scripts/`: data preparation, AI rewriting, training variant construction, format conversion, and checking scripts.
* `scripts/analysis/`: scripts for dataset statistics, topic words, linguistic features, and rewrite comparison.
* `results/dataset_analysis/`: selected summary-level dataset analysis files.
* `results/prediction_outputs/`: prediction output CSV files for different model variants and test conditions.
* `results/prediction_figures/`: figures showing prediction score and distance distributions.

---

## Experiment Pipeline

The experiment pipeline is:

1. Build clean authorship verification splits from Reddit data.
2. Generate AI-rewritten versions of selected training and test texts.
3. Construct different training variants.
4. Export the data into AdHominem-compatible format.
5. Train or evaluate AdHominem models.
6. Compare predictions across original and AI-rewritten test sets.
7. Analyze linguistic changes, prediction distributions, and distance distributions.

---

## Model Variants

The result files use the following model labels:

| Label | Meaning                                                    |
| ----- | ---------------------------------------------------------- |
| `O`   | Model trained on original data                             |
| `H`   | Model trained on hybrid data                               |
| `F`   | Model trained on fully rewritten or adjusted training data |

These labels appear in the files under:

```text
results/prediction_outputs/
results/prediction_figures/
```

Example files:

```text
O_orig.csv
O_Standard_rewrite.csv
O_Substantial_rewrite.csv
H_orig.csv
H_Standard_rewrite.csv
H_Substantial_rewrite.csv
F_orig.csv
F_Standard_rewrite.csv
F_Substantial_rewrite.csv
```

---

## Test Conditions

The experiments compare three types of test data:

| Condition             | Description                                |
| --------------------- | ------------------------------------------ |
| `orig`                | Original test texts                        |
| `Standard_rewrite`    | AI-rewritten texts with standard rewriting |
| `Substantial_rewrite` | AI-rewritten texts with stronger rewriting |

The goal is to evaluate whether AI rewriting affects authorship verification accuracy, model confidence, and robustness.

---

## Scripts

The main experiment scripts are stored in:

```text
scripts/
```

Important scripts include:

| Script                                    | Purpose                                       |
| ----------------------------------------- | --------------------------------------------- |
| `make_clean_splits.py`                    | Creates clean train/test splits               |
| `check_clean_splits.py`                   | Checks whether the clean splits are valid     |
| `rewrite_train_all.py`                    | Generates AI-rewritten training data          |
| `rewrite_test_sets.py`                    | Generates AI-rewritten test data              |
| `check_rewrite_files.py`                  | Checks whether rewrite outputs are complete   |
| `make_train_variants_from_rewrite_all.py` | Builds different training variants            |
| `check_train_variants.py`                 | Checks the constructed training variants      |
| `export_adh_format.py`                    | Exports data into AdHominem-compatible format |

Additional analysis scripts are stored in:

```text
scripts/analysis/
```

---

## Results

The `results/` directory contains selected result files from the thesis experiments.

```text
results/dataset_analysis/
```

Contains summary-level dataset analysis files, such as TF-IDF keywords, top words, and topic summaries.

```text
results/prediction_outputs/
```

Contains prediction output CSV files for different model variants and test conditions.

```text
results/prediction_figures/
```

Contains figures showing prediction score distributions and distance distributions.

---

## Data Availability and Ethical Notes

The full Reddit dataset and full AI-rewritten text data are not included in this repository.

The data used in this project contains user-generated social media text. Even when such text is publicly available, directly redistributing large amounts of user-generated content may raise privacy and ethical concerns. For this reason, this repository focuses on sharing the experiment code, selected summary results, prediction outputs, and analysis figures.

Large external files, such as pretrained embeddings and complete datasets, are also not included.

---

## Original AdHominem Model

The core model in this repository is based on the original AdHominem project:

**AdHominem: A tool for automatically analyzing the writing style in social media messages**

The original AdHominem model is an attention-based Siamese neural network for authorship verification in social media texts. It was proposed to learn neural features and visualize the decision-making process for authorship verification.

<img src="pic_attention.png" width="600">

The original repository contains the source code used in the paper:

[*Explainable Authorship Verification in Social Media via Attention-based Similarity Learning*](https://arxiv.org/abs/1910.08144), published at [*2019 IEEE International Conference on Big Data (IEEE BigData 2019)*](http://bigdataieee.org/BigData2019/).

---

## Installation

The original AdHominem project used Python 3.6 with Anaconda.

Main dependencies include:

* TensorFlow 1.12–1.15
* spaCy 2.3.2
* textacy 0.8.0
* fastText 0.9.2
* NumPy
* SciPy
* pandas
* scikit-learn
* bs4

Install dependencies using:

```bash
pip install -r requirements.txt
```

The project can also be run through the provided Dockerfile, depending on the local environment.

---

## Pretrained Word Embeddings

The original AdHominem model uses pretrained fastText word embeddings.

They can be downloaded from fastText:

```bash
cd data
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```

The embedding file is large and is not included in this repository.

---

## Running the Original AdHominem Pipeline

For the original AdHominem workflow, preprocessing can be run with:

```bash
python main_preprocess.py
```

Training can be started with:

```bash
python main_adhominem.py
```

In this thesis project, additional data preparation and rewriting steps are performed before exporting the data into AdHominem-compatible format.

---

## Citation and Acknowledgement

This project is based on the original AdHominem implementation. If you use the original model or code, please cite the original paper:

```bibtex
@inproceedings{Boenninghoff2019b,
  author={Benedikt Boenninghoff, Steffen Hessler, Dorothea Kolossa and Robert M. Nickel},
  title={Explainable Authorship Verification in Social Media via Attention-based Similarity Learning},
  booktitle={IEEE International Conference on Big Data (IEEE Big Data 2019), Los Angeles, CA, USA, December 9-12, 2019},
  year={2019}
}
```

Original project contact from the upstream README:

```text
benedikt.boenninghoff[at]rub.de
```

---

## Notes

This repository is intended mainly for documenting the experiment pipeline of my master's thesis. It is not a general-purpose cleaned release of AdHominem. The original AdHominem code remains the foundation of the model implementation, while the additional scripts and result files document the thesis-specific experimental workflow.
