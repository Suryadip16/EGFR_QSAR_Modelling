import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

data_df = pd.read_csv("final_model_data.csv")
x_df = data_df.drop(["Name", "Molecule_ChEMBL_ID", "pIC50"], axis=1)
y_df = data_df["pIC50"]
# x_df_small = x_df.iloc[0:100, ]
# y_df_small = y_df[0:100, ]
print(x_df.shape)
x = x_df.values
y = y_df.values

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

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
            nn.Linear(881, 80),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(80, 1),
        )

    def forward(self, x):
        return self.model(x)


# Initialize the model, loss function, and optimizer

model = FFN()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 5000
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
