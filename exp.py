import numpy as np

data_train = np.load("training_set.npy")
data_train = data_train.tolist()

print(data_train["data"][1])