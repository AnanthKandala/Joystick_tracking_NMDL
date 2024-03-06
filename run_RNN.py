import scipy.io
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
path = '/orange/physics-dept/an.kandala/coding_projects/Deep_learning_projects/Neuromatch_project/RNN'
os.chdir(path)
from data_loader import generate_data_loader
from model import CursorRNN
import pandas as pd


data_file = '/orange/physics-dept/an.kandala/coding_projects/Deep_learning_projects/Neuromatch_project/Dataset/data/fp_joystick.mat'
cleaned_data, train_data, test_data = generate_data_loader(data_file)


input_size = 60  # Number of features
hidden_size = 256  # Number of features in the hidden state
num_layers = 3  # Number of recurrent layers
output_size = 2  # Number of output dimensions (X and Y coordinates)
model = CursorRNN(input_size, hidden_size, num_layers, output_size).to(device)


# Choose loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# # Train the model
num_epochs = 100
losses = []  # To store the loss values

for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0.0
    for i in range(len(x_train)):
        inputs = x_train[i].unsqueeze(0).to(device)  # Add batch dimension
        targets = y_train[i].unsqueeze(0).to(device)  # Add batch dimension

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(x_train)  # Calculate average loss for the epoch
    losses.append(epoch_loss)  # Store the average loss

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')



fig, ax = plt.subplots(dpi=300)
ax.plot(losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss Across Training')
fig.savefig('training_loss.png')
torch.save(model.state_dict(), 'model.pth')