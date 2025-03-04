import pandas as pd
import os
import torch

data_file = os.path.join('../..', 'data', 'house_tiny.csv')
data = pd.read_csv(data_file)

print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)
print(outputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(y)