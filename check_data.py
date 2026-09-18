import numpy as np

# Target data
x1_target_train = np.load("Embeddings/x1_target_train.npy")
x2_target_train = np.load("Embeddings/x2_target_train.npy")
y_target_train = np.load("Embeddings/y_target_train.npy")

x1_target_test = np.load("Embeddings/x1_target_test.npy")
x2_target_test = np.load("Embeddings/x2_target_test.npy")
y_target_test = np.load("Embeddings/y_target_test.npy")

# Shadow data
x1_shadow_train = np.load("Embeddings/x1_shadow_train.npy")
x2_shadow_train = np.load("Embeddings/x2_shadow_train.npy")
y_shadow_train = np.load("Embeddings/y_shadow_train.npy")

x1_shadow_test = np.load("Embeddings/x1_shadow_test.npy")
x2_shadow_test = np.load("Embeddings/x2_shadow_test.npy")
y_shadow_test = np.load("Embeddings/y_shadow_test.npy")

print("TARGET TRAIN")
print("x1:", x1_target_train.shape)
print("x2:", x2_target_train.shape)
print("y :", y_target_train.shape)

print("\nTARGET TEST")
print("x1:", x1_target_test.shape)
print("x2:", x2_target_test.shape)
print("y :", y_target_test.shape)

print("\nSHADOW TRAIN")
print("x1:", x1_shadow_train.shape)
print("x2:", x2_shadow_train.shape)
print("y :", y_shadow_train.shape)

print("\nSHADOW TEST")
print("x1:", x1_shadow_test.shape)
print("x2:", x2_shadow_test.shape)
print("y :", y_shadow_test.shape)

print("\nLABEL COUNTS")
print("Target train:", np.unique(y_target_train, return_counts=True))
print("Target test :", np.unique(y_target_test, return_counts=True))
print("Shadow train:", np.unique(y_shadow_train, return_counts=True))
print("Shadow test :", np.unique(y_shadow_test, return_counts=True))