import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Make results more repeatable
np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# 1. LOAD SHADOW DATA
# =========================================================

x1_train = np.load("Embeddings/x1_shadow_train.npy")
x2_train = np.load("Embeddings/x2_shadow_train.npy")
y_train = np.load("Embeddings/y_shadow_train.npy")

x1_test = np.load("Embeddings/x1_shadow_test.npy")
x2_test = np.load("Embeddings/x2_shadow_test.npy")
y_test = np.load("Embeddings/y_shadow_test.npy")

print("Shadow data loaded")
print("Train:", x1_train.shape, x2_train.shape, y_train.shape)
print("Test :", x1_test.shape, x2_test.shape, y_test.shape)


# =========================================================
# 2. BUILD SIAMESE AUTOENCODER
# Adapted from sponsor-supplied notebook
# =========================================================

def build_siamese_autoencoder(embedding_dim):

    # Shared encoder
    encoder_input = Input(shape=(embedding_dim,))

    x = layers.Dense(
        50,
        activity_regularizer=regularizers.l1(0.01)
    )(encoder_input)

    x = layers.LeakyReLU(negative_slope=0.01)(x)

    encoder_output = layers.Dense(
        embedding_dim,
        activation="relu"
    )(x)

    encoder = Model(
        encoder_input,
        encoder_output,
        name="shadow_encoder"
    )

    # Decoder
    decoder_input = Input(shape=(embedding_dim,))

    decoder_output = layers.Dense(
        embedding_dim,
        activation="sigmoid"
    )(decoder_input)

    decoder = Model(
        decoder_input,
        decoder_output,
        name="shadow_decoder"
    )

    # Siamese inputs
    input1 = Input(shape=(embedding_dim,))
    input2 = Input(shape=(embedding_dim,))

    # IMPORTANT:
    # Both inputs use the SAME encoder
    encoded1 = encoder(input1)
    encoded2 = encoder(input2)

    recon1 = decoder(encoded1)
    recon2 = decoder(encoded2)

    merged_output = layers.Concatenate()(
        [recon1, recon2]
    )

    model = Model(
        inputs=[input1, input2],
        outputs=merged_output,
        name="shadow_siamese_autoencoder"
    )

    return model, encoder


# =========================================================
# 3. HYBRID LOSS
# Matches sponsor-supplied notebook
# =========================================================

def hybrid_classification_loss(margin=2.5, alpha=1.0):

    def loss_fn(y_true, y_pred):

        emb_dim = tf.shape(y_pred)[1] // 2

        recon1 = y_pred[:, :emb_dim]
        recon2 = y_pred[:, emb_dim:]

        recon_loss = tf.reduce_mean(
            tf.square(recon1 - recon2),
            axis=1
        )

        distances = tf.sqrt(
            tf.reduce_sum(
                tf.square(recon1 - recon2),
                axis=1
            )
        )

        y_true_float = tf.cast(y_true, tf.float32)

        contrastive_loss = (
            y_true_float * tf.square(distances)
            +
            (1 - y_true_float)
            * tf.square(
                tf.maximum(
                    margin - distances,
                    0
                )
            )
        )

        return tf.reduce_mean(
            alpha * recon_loss
            + contrastive_loss
        )

    return loss_fn


# =========================================================
# 4. TRAIN SIAMESE AUTOENCODER
# =========================================================

embedding_dim = x1_train.shape[1]

sa_model, encoder = build_siamese_autoencoder(
    embedding_dim
)

sa_model.compile(
    optimizer="adam",
    loss=hybrid_classification_loss(
        margin=2.5,
        alpha=1.0
    )
)

print("\nTraining Siamese autoencoder...")

sa_model.fit(
    [x1_train, x2_train],
    y_train,
    epochs=30,
    batch_size=256,
    validation_split=0.1
)


# =========================================================
# 5. CREATE ENCODED REPRESENTATIONS
# =========================================================

print("\nEncoding shadow records...")

encoded1_train = encoder.predict(
    x1_train,
    batch_size=256
)

encoded2_train = encoder.predict(
    x2_train,
    batch_size=256
)

encoded1_test = encoder.predict(
    x1_test,
    batch_size=256
)

encoded2_test = encoder.predict(
    x2_test,
    batch_size=256
)

# Difference between each record pair
diff_train = np.abs(
    encoded1_train - encoded2_train
)

diff_test = np.abs(
    encoded1_test - encoded2_test
)


# =========================================================
# 6. BUILD LINKAGE CLASSIFIER
# =========================================================

input_diff = Input(
    shape=(diff_train.shape[1],)
)

x = Dense(
    64,
    activation="relu"
)(input_diff)

x = Dense(
    32,
    activation="relu"
)(x)

output = Dense(
    1,
    activation="sigmoid"
)(x)

shadow_model = Model(
    inputs=input_diff,
    outputs=output,
    name="shadow_linkage_classifier"
)

shadow_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 7. TRAIN LINKAGE CLASSIFIER
# =========================================================

print("\nTraining shadow linkage classifier...")

shadow_model.fit(
    diff_train,
    y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.1
)


# =========================================================
# 8. EVALUATE SHADOW MODEL
# =========================================================

test_scores = shadow_model.predict(
    diff_test,
    batch_size=256
).flatten()

test_predictions = (
    test_scores >= 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

f1 = f1_score(
    y_test,
    test_predictions
)

print("\nSHADOW MODEL RESULTS")
print("--------------------")
print("Accuracy:", round(accuracy, 4))
print("F1 Score:", round(f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions
    )
)


# =========================================================
# 9. GENERATE DATA FOR FUTURE MIA
# =========================================================

member_scores = shadow_model.predict(
    diff_train,
    batch_size=256
).flatten()

nonmember_scores = shadow_model.predict(
    diff_test,
    batch_size=256
).flatten()

np.save(
    "Embeddings/shadow_member_scores.npy",
    member_scores
)

np.save(
    "Embeddings/shadow_nonmember_scores.npy",
    nonmember_scores
)

print("\nMEMBERSHIP DATA GENERATED")
print("-------------------------")
print("Members:", member_scores.shape)
print("Non-members:", nonmember_scores.shape)

print("\nShadow model training complete.")