import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score, f1_score

# -------------------------
# 1. Load shadow data
# -------------------------

x1_train = np.load("Embeddings/x1_shadow_train.npy")
x2_train = np.load("Embeddings/x2_shadow_train.npy")
y_train = np.load("Embeddings/y_shadow_train.npy")

x1_test = np.load("Embeddings/x1_shadow_test.npy")
x2_test = np.load("Embeddings/x2_shadow_test.npy")
y_test = np.load("Embeddings/y_shadow_test.npy")

print("Shadow data loaded.")
print("Train:", x1_train.shape, x2_train.shape, y_train.shape)
print("Test :", x1_test.shape, x2_test.shape, y_test.shape)


# -------------------------
# 2. Build Siamese encoder
# -------------------------

embedding_dim = x1_train.shape[1]

encoder_input = Input(shape=(embedding_dim,))

x = Dense(
    50,
    activity_regularizer=regularizers.l1(0.01)
)(encoder_input)

x = layers.LeakyReLU(negative_slope=0.01)(x)

encoder_output = Dense(
    embedding_dim,
    activation="relu"
)(x)

encoder = Model(
    encoder_input,
    encoder_output,
    name="encoder"
)


# -------------------------
# 3. Encode both summaries
# -------------------------

print("\nEncoding shadow data...")

encoded1_train = encoder.predict(x1_train, batch_size=256)
encoded2_train = encoder.predict(x2_train, batch_size=256)

encoded1_test = encoder.predict(x1_test, batch_size=256)
encoded2_test = encoder.predict(x2_test, batch_size=256)


# Difference between each pair
diff_train = np.abs(encoded1_train - encoded2_train)
diff_test = np.abs(encoded1_test - encoded2_test)


# -------------------------
# 4. Build linkage classifier
# -------------------------

input_diff = Input(shape=(diff_train.shape[1],))

x = Dense(64, activation="relu")(input_diff)
x = Dense(32, activation="relu")(x)

output = Dense(1, activation="sigmoid")(x)

shadow_model = Model(
    inputs=input_diff,
    outputs=output,
    name="shadow_linkage_model"
)

shadow_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -------------------------
# 5. Train
# -------------------------

print("\nTraining shadow linkage model...")

shadow_model.fit(
    diff_train,
    y_train,
    epochs=5,
    batch_size=256,
    validation_split=0.1
)


# -------------------------
# 6. Test
# -------------------------

probabilities = shadow_model.predict(
    diff_test,
    batch_size=256
).flatten()

predictions = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nSHADOW MODEL RESULTS")
print("--------------------")
print("Accuracy:", round(accuracy, 4))
print("F1 Score:", round(f1, 4))

# -------------------------
# 7. Save outputs for membership attack
# -------------------------

member_scores = shadow_model.predict(
    diff_train,
    batch_size=256
).flatten()

nonmember_scores = shadow_model.predict(
    diff_test,
    batch_size=256
).flatten()

np.save("Embeddings/shadow_member_scores.npy", member_scores)
np.save("Embeddings/shadow_nonmember_scores.npy", nonmember_scores)

print("\nSaved membership attack data.")
print("Member scores:", member_scores.shape)
print("Non-member scores:", nonmember_scores.shape)

print("Average member score:", member_scores.mean())
print("Average non-member score:", nonmember_scores.mean())

# -------------------------
# 8. Compare member vs non-member behaviour
# -------------------------

epsilon = 1e-7

# Confidence assigned to the CORRECT match/non-match class
member_confidence = np.where(
    y_train == 1,
    member_scores,
    1 - member_scores
)

nonmember_confidence = np.where(
    y_test == 1,
    nonmember_scores,
    1 - nonmember_scores
)

# Binary cross-entropy loss for each individual example
member_loss = -(
    y_train * np.log(member_scores + epsilon)
    + (1 - y_train) * np.log(1 - member_scores + epsilon)
)

nonmember_loss = -(
    y_test * np.log(nonmember_scores + epsilon)
    + (1 - y_test) * np.log(1 - nonmember_scores + epsilon)
)

print("\nMEMBERSHIP SIGNAL")
print("-----------------")

print("Average member confidence:",
      member_confidence.mean())

print("Average non-member confidence:",
      nonmember_confidence.mean())

print("Average member loss:",
      member_loss.mean())

print("Average non-member loss:",
      nonmember_loss.mean())