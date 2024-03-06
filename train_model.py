import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from data_loader import generate_data_loader
from model import CursorRNN
from torchsummary import summary

# Assuming x_train and y_train are tensors or numpy arrays containing your training data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 60  # Number of features
hidden_size = 1024  # Number of features in the hidden state
num_layers = 4  # Number of recurrent layers
output_size = 4  # Number of output dimensions (X and Y coordinates)
model = CursorRNN(input_size, hidden_size, num_layers, output_size).to(device)
params = 0
for p in model.parameters():
    params += p.numel()
print(f'Model has {params} parameters')
data_file = '/orange/physics-dept/an.kandala/coding_projects/Deep_learning_projects/Neuromatch_project/Dataset/data/fp_joystick.mat'
cleaned_data, train_loader, test_loader = generate_data_loader(data_file)
print(len(train_loader))

num_epochs = 100
losses = []  # To store the loss values

optimizer = Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed
criterion = MSELoss()  # Mean Squared Error loss

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)  # Calculate average loss for the epoch
    losses.append(epoch_loss)  # Store the average loss

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')

# Plot the loss
fig, ax = plt.subplots()
ax.plot(losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss Across Training')
fig.savefig('training_loss.png')

# Save the model
torch.save(model.state_dict(), 'model.pth')