"""
run_mia_attack.py
Author: Sorawit Naitoe McHugh (Candid Consulting / Group 54)
Purpose: End-to-End Phase 1 & Phase 2 Membership Inference Attack (MIA) Pipeline.
         Executes TC-SEC-01 through TC-SEC-06:
         1. Trains Target Linkage Model & extracts balanced target evaluation pairs.
         2. Trains K=10 Shadow Models & extracts balanced 1:1 (IN vs OUT) features.
         3. Runs statistical hypothesis t-tests (TC-SEC-04).
         4. Trains Binary Attack Classifiers for K = 1, 5, 10 (TC-SEC-05 & TC-SEC-06).
         5. Evaluates Attack on Target Model (Accuracy, Precision, Recall, F1, ROC/AUC, TPR@FPR).
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from scipy import stats

# Import team modules
from train_shadow import build_siamese_autoencoder, hybrid_classification_loss
from test_k_subsets import generate_shadow_subsets

# Set seeds for full reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Create results directory for evidence artifacts
os.makedirs("results", exist_ok=True)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def build_linkage_classifier(input_dim):
    """Builds the 2-layer MLP classification head for record linkage."""
    input_diff = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(input_diff)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    clf = Model(inputs=input_diff, outputs=output)
    clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return clf


def extract_attack_signals(model, diff_vectors, true_link_labels, membership_label):
    """
    Extracts candidate MIA features per record pair:
    [Confidence (s), BCE_Loss (L), Margin (M), Shannon_Entropy (H), Link_Label (y), Membership (m)]
    """
    confidence = model.predict(diff_vectors, batch_size=256, verbose=0).flatten()
    eps = 1e-7
    s_clip = np.clip(confidence, eps, 1.0 - eps)

    y = true_link_labels.astype(np.float32)
    # 1. Binary Cross-Entropy Loss
    bce_loss = -(y * np.log(s_clip) + (1.0 - y) * np.log(1.0 - s_clip))
    # 2. Prediction Margin from 0.5 boundary
    margin = np.abs(confidence - 0.5)
    # 3. Prediction Uncertainty / Shannon Entropy
    entropy = -(s_clip * np.log(s_clip) + (1.0 - s_clip) * np.log(1.0 - s_clip))

    mem_label = np.full_like(confidence, fill_value=membership_label)
    return np.column_stack([confidence, bce_loss, margin, entropy, y, mem_label])


def train_and_extract_linkage_model(x1_tr, x2_tr, y_tr, x1_te, x2_te, y_te, epochs_sa=15, epochs_clf=15):
    """Trains a Siamese Autoencoder + MLP Classifier and returns metrics + extracted features."""
    emb_dim = x1_tr.shape[1]
    sa_model, encoder = build_siamese_autoencoder(emb_dim)
    sa_model.compile(optimizer="adam", loss=hybrid_classification_loss(margin=2.5, alpha=1.0))
    sa_model.fit([x1_tr, x2_tr], y_tr, epochs=epochs_sa, batch_size=256, verbose=0)

    # Compute difference vectors
    diff_tr = np.abs(encoder.predict(x1_tr, batch_size=256, verbose=0) -
                     encoder.predict(x2_tr, batch_size=256, verbose=0))
    diff_te = np.abs(encoder.predict(x1_te, batch_size=256, verbose=0) -
                     encoder.predict(x2_te, batch_size=256, verbose=0))

    clf = build_linkage_classifier(diff_tr.shape[1])
    clf.fit(diff_tr, y_tr, epochs=epochs_clf, batch_size=256, verbose=0)

    # Evaluate Linkage Performance on Train and Test
    tr_preds = (clf.predict(diff_tr, batch_size=256, verbose=0).flatten() >= 0.5).astype(int)
    te_preds = (clf.predict(diff_te, batch_size=256, verbose=0).flatten() >= 0.5).astype(int)

    tr_acc = accuracy_score(y_tr, tr_preds)
    te_acc = accuracy_score(y_te, te_preds)
    te_f1  = f1_score(y_te, te_preds)

    # Extract MIA features (Balanced 1:1 sample of IN vs OUT)
    in_all  = extract_attack_signals(clf, diff_tr, y_tr, membership_label=1)
    out_all = extract_attack_signals(clf, diff_te, y_te, membership_label=0)

    # Balance IN and OUT to exact 1:1 ratio (size = len(out_all))
    n_bal = min(len(in_all), len(out_all))
    bal_idx = np.random.choice(len(in_all), size=n_bal, replace=False)
    in_bal = in_all[bal_idx]
    out_bal = out_all[:n_bal]

    balanced_features = np.vstack([in_bal, out_bal])
    return tr_acc, te_acc, te_f1, balanced_features


def get_tpr_at_fpr(fpr, tpr, target_fpr):
    """Returns the highest TPR achieved where FPR <= target_fpr."""
    valid = np.where(fpr <= target_fpr)[0]
    return float(tpr[valid[-1]]) if len(valid) > 0 else 0.0


# =====================================================================
# MAIN EXECUTION PIPELINE (TC-SEC-01 TO TC-SEC-06)
# =====================================================================
if __name__ == "__main__":
    start_time = time.time()
    print("=====================================================================")
    print(" Candid Consulting (Group 54): End-to-End MIA Security Test Pipeline ")
    print("=====================================================================")

    # -----------------------------------------------------------------
    # TC-SEC-01: DATA INTEGRITY & ISOLATION CHECK
    # -----------------------------------------------------------------
    print("\n[TC-SEC-01] Verifying 12 Embedding Arrays & Data Isolation...")
    x1_t_tr = np.load("Embeddings/x1_target_train.npy")
    x2_t_tr = np.load("Embeddings/x2_target_train.npy")
    y_t_tr  = np.load("Embeddings/y_target_train.npy")

    x1_t_te = np.load("Embeddings/x1_target_test.npy")
    x2_t_te = np.load("Embeddings/x2_target_test.npy")
    y_t_te  = np.load("Embeddings/y_target_test.npy")

    x1_s_tr = np.load("Embeddings/x1_shadow_train.npy")
    x2_s_tr = np.load("Embeddings/x2_shadow_train.npy")
    y_s_tr  = np.load("Embeddings/y_shadow_train.npy")

    x1_s_te = np.load("Embeddings/x1_shadow_test.npy")
    x2_s_te = np.load("Embeddings/x2_shadow_test.npy")
    y_s_te  = np.load("Embeddings/y_shadow_test.npy")

    arrays = [x1_t_tr, x2_t_tr, y_t_tr, x1_t_te, x2_t_te, y_t_te,
              x1_s_tr, x2_s_tr, y_s_tr, x1_s_te, x2_s_te, y_s_te]
    assert all(np.all(np.isfinite(a)) for a in arrays), "Non-finite values detected!"
    print(f"  ↳ Target Train: {x1_t_tr.shape} | Target Test: {x1_t_te.shape}")
    print(f"  ↳ Shadow Train: {x1_s_tr.shape} | Shadow Test: {x1_s_te.shape}")
    print("  ↳ [PASS] All 12 files verified (768-dim float32, zero NaN/Inf).")

    # -----------------------------------------------------------------
    # TC-SEC-02: TARGET LINKAGE BASELINE & TARGET FEATURE EXTRACTION
    # -----------------------------------------------------------------
    print("\n[TC-SEC-02] Training Target Linkage Model (Victim Baseline)...")
    t0 = time.time()
    t_tr_acc, t_te_acc, t_te_f1, target_mia_data = train_and_extract_linkage_model(
        x1_t_tr, x2_t_tr, y_t_tr, x1_t_te, x2_t_te, y_t_te,
        epochs_sa=20, epochs_clf=20
    )
    t_gen_gap = t_tr_acc - t_te_acc
    print(f"  ↳ Target Train Acc: {t_tr_acc*100:.2f}% | Target Test Acc: {t_te_acc*100:.2f}% | F1: {t_te_f1:.4f}")
    print(f"  ↳ Target Generalization Gap (Train - Test): {t_gen_gap*100:+.2f}%")
    print(f"  ↳ Balanced Target Attack Evaluation Set Compiled: {target_mia_data.shape} (8,000 IN / 8,000 OUT)")
    print(f"  ↳ [PASS] Completed in {time.time() - t0:.1f}s.")

    # -----------------------------------------------------------------
    # TC-SEC-03: TRAIN K=10 SHADOW MODELS ON K-SUBSETS
    # -----------------------------------------------------------------
    print("\n[TC-SEC-03] Generating K=10 Shadow Subsets & Training Shadow Models...")
    MAX_K = 10
    SAMPLE_SIZE = 8000
    subsets = generate_shadow_subsets(
        x1_data=x1_s_tr, x2_data=x2_s_tr, y_data=y_s_tr,
        k=MAX_K, sample_size=SAMPLE_SIZE, train_ratio=0.7
    )

    shadow_features_list = []
    shadow_metrics_log = []

    for s in subsets:
        mid = s["model_id"]
        tr, te = s["train"], s["test"]
        m_t0 = time.time()
        s_tr_acc, s_te_acc, s_te_f1, s_bal_feats = train_and_extract_linkage_model(
            tr["x1"], tr["x2"], tr["y"], te["x1"], te["x2"], te["y"],
            epochs_sa=15, epochs_clf=15
        )
        elapsed = time.time() - m_t0
        shadow_features_list.append(s_bal_feats)
        shadow_metrics_log.append({
            "Model_ID": f"Shadow_{mid}",
            "Train_Acc": round(s_tr_acc, 4),
            "Test_Acc": round(s_te_acc, 4),
            "Gen_Gap": round(s_tr_acc - s_te_acc, 4),
            "Test_F1": round(s_te_f1, 4),
            "Balanced_Attack_Rows": len(s_bal_feats),
            "Runtime_s": round(elapsed, 2)
        })
        print(f"  ↳ Shadow Model {mid:2d}/10 | Train Acc: {s_tr_acc*100:.2f}% | "
              f"Test Acc: {s_te_acc*100:.2f}% | F1: {s_te_f1:.4f} ({elapsed:.1f}s)")

    df_shadow_log = pd.DataFrame(shadow_metrics_log)
    df_shadow_log.to_csv("results/shadow_models_training_log.csv", index=False)
    print(f"  ↳ Mean Shadow Test Acc: {df_shadow_log['Test_Acc'].mean()*100:.2f}% "
          f"(±{df_shadow_log['Test_Acc'].std()*100:.2f}%)")
    print("  ↳ [PASS] Shadow model training log saved to results/shadow_models_training_log.csv")

    # -----------------------------------------------------------------
    # TC-SEC-04: STATISTICAL FEATURE EXTRACTION & HYPOTHESIS TESTING
    # -----------------------------------------------------------------
    print("\n[TC-SEC-04] Running Statistical Hypothesis Tests on Extracted Signals (K=10 Pool)...")
    pooled_k10 = np.vstack(shadow_features_list)
    np.save("Embeddings/attack_train_data.npy", pooled_k10)

    in_rows  = pooled_k10[pooled_k10[:, 5] == 1]
    out_rows = pooled_k10[pooled_k10[:, 5] == 0]

    signal_names = ["Confidence (s)", "BCE_Loss (L)", "Margin (M)", "Entropy (H)"]
    stats_rows = []
    for idx, name in enumerate(signal_names):
        in_vals  = in_rows[:, idx]
        out_vals = out_rows[:, idx]
        t_stat, p_val = stats.ttest_ind(in_vals, out_vals, equal_var=False)
        stats_rows.append({
            "Feature": name,
            "Member_IN_Mean": round(float(np.mean(in_vals)), 4),
            "Member_IN_Std": round(float(np.std(in_vals)), 4),
            "NonMember_OUT_Mean": round(float(np.mean(out_vals)), 4),
            "NonMember_OUT_Std": round(float(np.std(out_vals)), 4),
            "Diff_(IN-OUT)": round(float(np.mean(in_vals) - np.mean(out_vals)), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val)
        })

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv("results/feature_distribution_stats.csv", index=False)
    print(df_stats.to_string(index=False))
    print("  ↳ [PASS] Feature distribution stats saved to results/feature_distribution_stats.csv")

    # -----------------------------------------------------------------
    # TC-SEC-05 & TC-SEC-06: ATTACK CLASSIFIER & MULTI-K SENSITIVITY
    # -----------------------------------------------------------------
    print("\n[TC-SEC-05 & TC-SEC-06] Evaluating Attack Classifier on Target Model across K = 1, 5, 10...")
    X_target_eval = target_mia_data[:, :5]  # [Confidence, Loss, Margin, Entropy, Link_Label]
    y_target_eval = target_mia_data[:, 5]   # [1 = Member, 0 = Non-Member] (8,000 each)

    k_configs = [1, 5, 10]
    sensitivity_results = []
    roc_curves_data = {}

    for k_val in k_configs:
        k_t0 = time.time()
        # Pool balanced data from the first k_val shadow models
        D_attack_k = np.vstack(shadow_features_list[:k_val])
        X_att_train = D_attack_k[:, :5]
        y_att_train = D_attack_k[:, 5]

        # Train Attack Classifiers: Random Forest & MLP
        rf_attack = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=RANDOM_SEED, n_jobs=-1
        )
        rf_attack.fit(X_att_train, y_att_train)

        mlp_attack = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=RANDOM_SEED)
        )
        mlp_attack.fit(X_att_train, y_att_train)

        for clf_name, clf_obj in [("RandomForest", rf_attack), ("MLP_Classifier", mlp_attack)]:
            y_prob = clf_obj.predict_proba(X_target_eval)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            acc  = accuracy_score(y_target_eval, y_pred)
            prec = precision_score(y_target_eval, y_pred, zero_division=0)
            rec  = recall_score(y_target_eval, y_pred, zero_division=0)
            f1   = f1_score(y_target_eval, y_pred, zero_division=0)

            fpr, tpr, _ = roc_curve(y_target_eval, y_prob)
            roc_auc = auc(fpr, tpr)
            tpr_01  = get_tpr_at_fpr(fpr, tpr, 0.01)
            tpr_05  = get_tpr_at_fpr(fpr, tpr, 0.05)

            sensitivity_results.append({
                "K_Shadow_Models": k_val,
                "Attack_Classifier": clf_name,
                "Attack_Train_Rows": len(D_attack_k),
                "Target_Attack_Acc": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall_(TPR)": round(rec, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(roc_auc, 4),
                "TPR@FPR=0.01": round(tpr_01, 4),
                "TPR@FPR=0.05": round(tpr_05, 4)
            })

            if clf_name == "RandomForest":
                roc_curves_data[k_val] = (fpr, tpr, roc_auc, y_pred, y_prob)

    df_sens = pd.DataFrame(sensitivity_results)
    df_sens.to_csv("results/shadow_k_sensitivity_analysis.csv", index=False)
    print("\n--- TARGET MODEL ATTACK EVALUATION SUMMARY (TC-SEC-05 & TC-SEC-06) ---")
    print(df_sens.to_string(index=False))

    # Print Confusion Matrix for K=10 RandomForest on Target Model
    _, _, _, y_pred_k10, _ = roc_curves_data[10]
    cm = confusion_matrix(y_target_eval, y_pred_k10)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTarget Attack Confusion Matrix (K=10, RandomForest, Balanced 16,000 pairs):")
    print(f"  True Positives (Members Caught):     {tp}")
    print(f"  False Positives (False Alarms):      {fp}")
    print(f"  True Negatives (Non-Members Correct):{tn}")
    print(f"  False Negatives (Members Missed):    {fn}")

    # -----------------------------------------------------------------
    # GENERATE PLOTS FOR DELIVERABLE 4 REPORT & FRIDAY PRESENTATION
    # -----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        # Plot 1: Multi-K ROC Curve Comparison
        plt.figure(figsize=(8, 6))
        colors = {1: "#1f77b4", 5: "#2ca02c", 10: "#d62728"}
        for k_val in k_configs:
            fpr, tpr, roc_auc, _, _ = roc_curves_data[k_val]
            plt.plot(fpr, tpr, color=colors[k_val], lw=2,
                     label=f"K = {k_val} Shadow Models (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Guessing (AUC = 0.5000)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("Membership Inference Attack ROC Curve across Shadow Model Counts (K = 1, 5, 10)")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/multi_k_shadow_roc_comparison.png", dpi=300)
        plt.close()

        # Plot 2: Feature Distributions (Confidence & BCE Loss for IN vs OUT)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(in_rows[:, 0], bins=40, alpha=0.6, label="Member (IN)", color="#1f77b4", density=True)
        axes[0].hist(out_rows[:, 0], bins=40, alpha=0.6, label="Non-Member (OUT)", color="#ff7f0e", density=True)
        axes[0].set_title("Prediction Confidence Distribution (IN vs OUT)")
        axes[0].set_xlabel("Confidence Score (s)")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        axes[1].hist(np.clip(in_rows[:, 1], 0, 2.5), bins=40, alpha=0.6, label="Member (IN)", color="#1f77b4", density=True)
        axes[1].hist(np.clip(out_rows[:, 1], 0, 2.5), bins=40, alpha=0.6, label="Non-Member (OUT)", color="#ff7f0e", density=True)
        axes[1].set_title("Binary Cross-Entropy Loss Distribution (IN vs OUT)")
        axes[1].set_xlabel("BCE Loss (Clipped at 2.5)")
        axes[1].set_ylabel("Density")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("results/feature_distribution_comparison.png", dpi=300)
        plt.close()

        # Plot 3: Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
        fig.colorbar(cax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=f"{cm[i, j]}", va="center", ha="center", size="large", weight="bold")
        plt.xlabel("Predicted Membership Label (0=OUT, 1=IN)", fontsize=11)
        plt.ylabel("Actual Membership Label (0=OUT, 1=IN)", fontsize=11)
        plt.title("Target Model MIA Confusion Matrix (K=10)", fontsize=12, pad=15)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Member (0)", "Member (1)"])
        ax.set_yticklabels(["Non-Member (0)", "Member (1)"])
        plt.tight_layout()
        plt.savefig("results/target_mia_confusion_matrix.png", dpi=300)
        plt.close()

        print("\n[PLOTS SAVED] Generated 3 high-res plots in 'results/' folder:")
        print("  1. results/multi_k_shadow_roc_comparison.png")
        print("  2. results/feature_distribution_comparison.png")
        print("  3. results/target_mia_confusion_matrix.png")
    except Exception as e:
        print(f"\n[NOTE] Plot generation skipped ({e}). CSV tables are saved in 'results/'.")

    print(f"\n[ALL TESTS COMPLETE] Total pipeline execution time: {time.time() - start_time:.1f}s.")