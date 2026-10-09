"""
train_multi_shadow.py
Author: Sorawit Naitoe McHugh & Jake Barbouttis (Group 54)
Purpose: Multi-shadow model training loop & attack feature extraction
         for Phase 1 Membership Inference Attack against Record Linkage.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, f1_score

# =====================================================================
# 1. MODULAR IMPORTS FROM TEAM SCRIPTS
# =====================================================================
# Import Jake's Siamese Autoencoder & Loss from train_shadow.py
from train_shadow import build_siamese_autoencoder, hybrid_classification_loss

# Import your K-subset data splitting function from test_k_subsets.py
from test_k_subsets import generate_shadow_subsets

# Set random seeds for consistent results
np.random.seed(42)
tf.random.set_seed(42)


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================
def build_linkage_classifier(input_dim):
    """
    Builds the 2-layer MLP classification head to evaluate record linkage.
    """
    input_diff = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(input_diff)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    
    clf = Model(inputs=input_diff, outputs=output, name="linkage_head")
    clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return clf


def extract_attack_signals(model, diff_vectors, true_link_labels, membership_label):
    """
    Extracts the 3 core security features defined in Section 2.1 of our Scoping Doc:
    - Confidence Score (s)
    - Binary Cross-Entropy Loss (L)
    - Prediction Margin (M = |s - 0.5|)
    """
    # 1. Prediction confidence s in [0.0, 1.0]
    confidence = model.predict(diff_vectors, batch_size=256, verbose=0).flatten()

    # Clip values slightly to prevent log(0) numerical errors
    eps = 1e-7
    s_clipped = np.clip(confidence, eps, 1.0 - eps)

    # 2. Binary Cross-Entropy Loss (L)
    y = true_link_labels.astype(np.float32)
    bce_loss = -(y * np.log(s_clipped) + (1.0 - y) * np.log(1.0 - s_clipped))

    # 3. Prediction Margin (M)
    margin = np.abs(confidence - 0.5)

    # Membership Label (1 for IN / Training Member, 0 for OUT / Non-Member)
    mem_label = np.full_like(confidence, fill_value=membership_label)

    # Stacks into matrix: [confidence, loss, margin, link_label, membership_label]
    return np.column_stack([confidence, bce_loss, margin, y, mem_label])


# =====================================================================
# 3. MAIN TRAINING & EXTRACTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    print("=================================================================")
    print(" Candid Consulting: Phase 1 Multi-Shadow Model Training Pipeline ")
    print("=================================================================")

    # 1. Load Shadow Embeddings
    print("\nLoading shadow dataset embeddings...")
    x1_shadow = np.load("Embeddings/x1_shadow_train.npy")
    x2_shadow = np.load("Embeddings/x2_shadow_train.npy")
    y_shadow  = np.load("Embeddings/y_shadow_train.npy")

    # 2. Configure K and Sample Size
    # We use K = 3 for quick laptop execution (~3-4 mins total).
    # You can change to K = 5 or K = 10 later for Deliverable 4!
    K_MODELS = 3
    SAMPLE_SIZE = 8000
    TRAIN_RATIO = 0.7  # 70% train (5,600 pairs), 30% test (2,400 pairs)

    subsets = generate_shadow_subsets(
        x1_data=x1_shadow,
        x2_data=x2_shadow,
        y_data=y_shadow,
        k=K_MODELS,
        sample_size=SAMPLE_SIZE,
        train_ratio=TRAIN_RATIO
    )

    embedding_dim = x1_shadow.shape[1]
    all_attack_records = []
    shadow_model_results = []

    # 3. Train Each Shadow Model
    for s in subsets:
        model_id = s["model_id"]
        train_data = s["train"]
        test_data  = s["test"]

        print(f"\n>>> [Shadow Model {model_id}/{K_MODELS}] Training on {len(train_data['y'])} pairs...")

        # Step A: Train Siamese Autoencoder (15 epochs is fast and reaches good convergence)
        sa_model, encoder = build_siamese_autoencoder(embedding_dim)
        sa_model.compile(optimizer="adam", loss=hybrid_classification_loss(margin=2.5, alpha=1.0))
        
        sa_model.fit(
            [train_data["x1"], train_data["x2"]],
            train_data["y"],
            epochs=15,
            batch_size=256,
            verbose=0  # Keeps the terminal clean
        )

        # Step B: Encode representations and compute absolute difference vectors
        enc1_tr = encoder.predict(train_data["x1"], batch_size=256, verbose=0)
        enc2_tr = encoder.predict(train_data["x2"], batch_size=256, verbose=0)
        diff_train = np.abs(enc1_tr - enc2_tr)

        enc1_te = encoder.predict(test_data["x1"], batch_size=256, verbose=0)
        enc2_te = encoder.predict(test_data["x2"], batch_size=256, verbose=0)
        diff_test  = np.abs(enc1_te - enc2_te)

        # Step C: Train Linkage Classifier Head (15 epochs)
        shadow_clf = build_linkage_classifier(diff_train.shape[1])
        shadow_clf.fit(
            diff_train,
            train_data["y"],
            epochs=15,
            batch_size=256,
            verbose=0
        )

        # Step D: Evaluate linkage accuracy on the shadow test split
        test_preds = (shadow_clf.predict(diff_test, batch_size=256, verbose=0).flatten() >= 0.5).astype(int)
        acc = accuracy_score(test_data["y"], test_preds)
        f1  = f1_score(test_data["y"], test_preds)
        
        print(f"    ↳ Linkage Test Accuracy: {acc*100:.2f}% | F1 Score: {f1:.4f}")
        shadow_model_results.append((model_id, acc, f1))

        # Step E: Extract Attack Features for Phase 2
        # Training data = IN (Member / 1)
        in_features = extract_attack_signals(shadow_clf, diff_train, train_data["y"], membership_label=1)
        # Testing data = OUT (Non-Member / 0)
        out_features = extract_attack_signals(shadow_clf, diff_test, test_data["y"], membership_label=0)

        all_attack_records.append(in_features)
        all_attack_records.append(out_features)

    # 4. Compile and Save the Attack Dataset for Phase 2
    master_attack_data = np.vstack(all_attack_records)
    output_path = "Embeddings/attack_train_data.npy"
    np.save(output_path, master_attack_data)

    # 5. Print Final Summary Table
    print("\n" + "=" * 65)
    print(" PHASE 1 SHADOW TRAINING SUMMARY ")
    print("=" * 65)
    print(f"{'Model ID':<15} | {'Test Linkage Accuracy':<25} | {'F1 Score':<12}")
    print("-" * 65)
    for mid, acc, f1 in shadow_model_results:
        print(f"Shadow Model {mid:<2} | {acc*100:.2f}%{'':<18} | {f1:.4f}")

    print("\n" + "=" * 65)
    print(" ATTACK DATASET EXTRACTED FOR PHASE 2 ")
    print("=" * 65)
    print(f"Saved to: {output_path}")
    print(f"Total Observations: {master_attack_data.shape[0]} rows")
    print(f"Columns: [Confidence, BCE_Loss, Margin, Link_Label, Membership_Label]")

    members = int(np.sum(master_attack_data[:, 4] == 1))
    non_members = int(np.sum(master_attack_data[:, 4] == 0))
    print(f"Label Balance: {members} Members (IN) / {non_members} Non-Members (OUT)")
    print("\n[SUCCESS] Phase 1 is officially complete. Ready for Phase 2 Attack Model training!")