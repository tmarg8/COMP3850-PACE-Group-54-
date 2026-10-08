"""
test_k_subsets.py
Author: Sorawit Naitoe McHugh (Candid Consulting / Group 54)
Purpose: Implementation and verification of K-subset data partitioning
         for Phase 1 Shadow Model Training (resolving 23/09 sponsor feedback)
"""

import os
import numpy as np

# Set specific seed for reproducible sampling
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

### STEP 1: THE K-SUBSET SAMPLING FUNCTION
def generate_shadow_subsets(
        x1_data,
        x2_data,
        y_data,
        k=5,
        sample_size=8000, # Sweet spot is 8,000 - 10,000
        train_ratio=0.7 # Ratio value mirroring Dinusha's presentation slides
):
    """
    Samples K subsets from the shadow pool to train K shadow models.

    Parameters:
        x1_data: ClinicalBERT embeddings for patient note 1
        x2_data: ClinicalBERT embeddings for patient note 2
        y_data: Linkage match labels (1.0 or 0.0)
        k (int): Number of shadow models to train (set to 5 if unspecified)
        sample_size (int): Number of record pairs per shadow model (set to 8,000 if unspecified)
        train_ratio (float): Ratio of pairs used for training vs testing (set to 0.7 if unspecified)

    Returns:
        list of dict: K distinct dictionary subsets containing train and test partitions
    """
    total_records = len(y_data)
    subsets = []

    n_train = int(sample_size * train_ratio) # 8000 * 0.7 = 5,600 pairs to train
    n_test = sample_size - n_train          # 8000 - 5600 = 2,400 pairs to test

    print(f"\n--- GENERATING {k} SHADOW SUBSETS ---")
    print(f"Total Shadow Pool: {total_records} pairs")
    print(f"Target Size per Shadow Model: {sample_size} pairs")
    print(f" ↳ Train Pairs (Members / IN):     {n_train}")
    print(f" ↳ Test Pairs (Non-Members / OUT): {n_test}")

    for model_idx in range(1, k + 1):
        # 1. Randomly sample sample_size indices from the pool (with overlap across models)
        # Using stratified sampling to ensure 50/50 match/non-match balance
        
        match_idx = np.where(y_data == 1.0)[0] # Finds the row numbers of all true matches
                                               # (where two notes belong to the same person)
        
        nonmatch_idx = np.where(y_data == 0.0)[0] # Finds the row numbers of all non-matches (diff ppl)

        half_sample = sample_size // 2 # 4,000 matches and 4,000 non-matches

        # Randomly draws 4,000 match rows and 4,000 non-match rows
        sampled_matches = np.random.choice(match_idx, size=half_sample, replace=False)
        sampled_nonmatches = np.random.choice(nonmatch_idx, size=half_sample, replace=False)

        # Combine and shuffle
        all_sampled = np.concatenate([sampled_matches, sampled_nonmatches])
        np.random.shuffle(all_sampled)

        # 2. Split into disjoint Train and Test partitions for this specific shadow model

        # Slices the 8,000 row numbers into...
        train_indices = all_sampled[:n_train] # 5,600 row numbers assigned to Training
        test_indices = all_sampled[n_train:] # 2,400 row numbers assigned to Testing

        # 3. Package the data neatly into a Python dictionary
        subset = {
            "model_id": model_idx,
            "train": {
                "x1": x1_data[train_indices],
                "x2": x2_data[train_indices],
                "y": y_data[train_indices]
            },
            "test": {
                "x1": x1_data[test_indices],
                "x2": x2_data[test_indices],
                "y": y_data[test_indices]
            }
        }
        subsets.append(subset)

    print(f"Successfully generated {len(subsets)} shadow model partitions. \n")
    return subsets

### STEP 2: RUN IT AND TEST
if __name__ == "__main__":
    print("Loading shadow embeddings...")
    x1_shadow = np.load("Embeddings/x1_shadow_train.npy")
    x2_shadow = np.load("Embeddings/x2_shadow_train.npy")
    y_shadow = np.load("Embeddings/y_shadow_train.npy")

    # Generate 5 shadow subsets
    shadow_subsets = generate_shadow_subsets(
        x1_data=x1_shadow,
        x2_data=x2_shadow,
        y_data=y_shadow,
        k=5,
        sample_size=8000,
        train_ratio=0.7
    )