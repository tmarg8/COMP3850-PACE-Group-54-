# COMP3850 PACE Group 54

This project explores a membership inference attack (MIA) against a record linkage model. The goal is to evaluate whether an attacker can infer whether a record pair was part of a model's training data by analysing model behaviour and output signals.

## Project overview

The repository is built around a Python pipeline that:
- loads and validates embedding data,
- trains shadow and target linkage models,
- extracts membership-related signals such as confidence, loss, and margin,
- evaluates attack performance across multiple shadow-model setups,
- saves results and plots for analysis.

## File structure and purpose

### `check_data.py`
Checks the integrity and shape of the training and testing embedding arrays. It prints dataset dimensions and label counts for both the target and shadow datasets, helping confirm that the data is loaded correctly and balanced.

### `train_shadow.py`
Trains a single shadow linkage model using a Siamese autoencoder plus a classification head. It encodes record pairs into embeddings, compares their differences, and trains a binary linkage classifier to distinguish matching and non-matching pairs.

### `train_multi_shadow.py`
Extends the single-shadow workflow into a multi-shadow training pipeline. It creates multiple shadow-model subsets, trains them independently, and extracts attack feature vectors for downstream membership inference analysis.

### `test_k_subsets.py`
Defines the K-subset sampling logic used to partition the shadow dataset into multiple training/test subsets. This allows experimenters to test how the number of shadow models affects attack performance.

### `run_mia_attack.py`
Runs the end-to-end attack pipeline. It trains the target linkage model, builds shadow models, extracts attack signals, evaluates a membership inference classifier, and saves findings such as confusion matrices, ROC curves, and CSV summaries in the `results/` folder.

### `results/`
Stores the generated evidence and evaluation outputs from the attack pipeline, including:
- CSV summaries of model metrics,
- feature distribution statistics,
- shadow-model sensitivity results,
- plots for ROC and confusion-matrix comparisons.

### `.gitignore`
Excludes local environment and generated data files such as virtual environments, caches, and large numpy arrays.

## Discussion of findings

##

