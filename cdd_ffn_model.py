import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv("final_model_data.csv")
# print(data.columns)
data.drop(["Molecule_ChEMBL_ID"], inplace=True, axis=1)
# print(data.columns)

# Make and X and Y Matrix

x = data.drop(["Name", "pIC50"], axis=1)
# print(data_x.columns)

selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
data_x = selection.fit_transform(x)
print(data_x.shape)

data_y = data["pIC50"]
# x_df_small = x_df.iloc[0:100, ]
# y_df_small = y_df[0:100, ]
# print(x_df.shape)
# x = data_x.values
# y = data_y.values

x_tensor = torch.tensor(data_x, dtype=torch.float32)
y_tensor = torch.tensor(data_y, dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.3, shuffle=True, random_state=42)

# Create TensorDataset for train and test sets
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Create Dataloader for batching
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, num_workers=12, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, num_workers=12, shuffle=False)


# Model Architecture:
class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(144, 108),
            nn.Tanh(),
            nn.Linear(108, 72),
            nn.Tanh(),
            nn.Linear(72, 36),
            nn.Tanh(),
            nn.Linear(36, 18),
            nn.Tanh(),
            nn.Linear(18, 1)
        )

    def forward(self, x):
        return self.model(x)


# Initialize the model, loss function, and optimizer

model = FFN()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch, (features, target) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Forward Pass
        pred = model(features).squeeze(1)
        loss = loss_function(pred, target)

        # Backprop and Optimization:
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader): .4f}")

# Eval On test set:
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for features, target in test_dataloader:
        model_pred = model(features)
        all_preds.append(model_pred.numpy())
        all_true.append(target.numpy())

# Concatenate predictions and targets from diff batches
all_preds = np.concatenate(all_preds, axis=0)
all_true = np.concatenate(all_true, axis=0)

# Calculate R-squared and Mean Absolute Error
r2 = r2_score(all_true, all_preds)
mae = mean_absolute_error(all_true, all_preds)

print(f'Test R-squared: {r2: .4f}')
print(f'Test Mean Absolute Error: {mae: .4f}')
