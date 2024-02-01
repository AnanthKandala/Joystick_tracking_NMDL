import scipy.io
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_images(data_file, im_len, stride_len):
    '''takes the matlab file for a patient and creates images of the ecog data for training/testing. We have a 2D array made up of
    electrode voltage vs time. This single, long signal needs to chopped up into smaller pieces to create the desired dataset of 'images'
    args:
        data_file: matlab file for a patient
        i_len (int): length of each image (in ms)
        stride_len (int): length of stride (in ms)
    returns:
        image_data (list): list of (image, label) pairs. image (2D array) -->ecog voltages vs time, label (2D array) --> joystick coordinates vs time '''
    data = data_file
    array = data['data']
    array = array.astype(np.float32)
    indices = [(i, i+im_len) for i in list(range(0, array.shape[0] - im_len +1, stride_len))]
    cut_signals = [array[i[0]:i[1], :] for i in indices]
    assert len(data['CursorPosX']) == len(data['CursorPosY'])
    t_sample = [i+im_len-1 for i in list(range(0, len(data['CursorPosX']) - im_len +1, stride_len))]
    trajectory = [(data['CursorPosX'][i,0], data['CursorPosY'][i,0]) for i in t_sample]
    return [(cut_signals[i], trajectory[i]) for i in range(len(cut_signals))]

# Load data
dataDir = r'C:/Users/jeffr/PycharmProjects/EEG_RNN/joystick_track/data'
files = os.listdir(dataDir)
data_file = scipy.io.loadmat(os.path.join(dataDir, files[3])) #Index 0 has 60 inputs, 1 and 2 have 64, and 3 has 48

# Create (image, label) pairs using create_images function
im_len = 1000  # Length of each image in ms (adjust to your desired value)
stride_len = 100  # Stride length in ms (adjust to your desired value)
image_data = create_images(data_file, im_len, stride_len)

#Extract and store data in x_train and y_train objects. Use as batch size of 612 for training.
batch_size = 207
x_train = torch.stack([torch.tensor(pair[0], dtype=torch.float32) for pair in image_data])
y_train = torch.tensor([list(pair[1]) for pair in image_data], dtype=torch.float32)

#Data normalization
# Find the minimum and maximum values for x and y coordinates
x_min = y_train[:, 0].min()
x_max = y_train[:, 0].max()
y_min = y_train[:, 1].min()
y_max = y_train[:, 1].max()

# Define the normalization function
def normalize_coordinates(y):
    y[:, 0] = (y[:, 0] - x_min) / (x_max - x_min)
    y[:, 1] = (y[:, 1] - y_min) / (y_max - y_min)
    return y

# Normalize the x and y coordinates within y_train
normalized_y_train = normalize_coordinates(y_train.clone())
y_train = normalized_y_train

test_ratio = 0.2
test_size = int(len(x_train) * test_ratio)
train_size = len(x_train) - test_size

class ImageLabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_data):
        self.image_data = image_data

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image, label = self.image_data[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Now, use the custom dataset class to create train_dataset and test_dataset
train_dataset = ImageLabelDataset(image_data[:train_size])
test_dataset = ImageLabelDataset(image_data[train_size:])

# Create data loaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Normalize the x and y coordinates in the test dataset using the same normalization function
# Normalize the x and y coordinates in the test dataset using the same normalization function
def normalize_test_data(data, x_min, x_max, y_min, y_max):
    # Reshape the 1D tensor to 2D with a single row
    data = data.unsqueeze(0)
    data[:, 0] = (data[:, 0] - x_min) / (x_max - x_min)
    data[:, 1] = (data[:, 1] - y_min) / (y_max - y_min)
    return data.squeeze(0)  # Squeeze the tensor to remove the extra dimension

# Normalize the test data using the same normalization parameters from y_train
test_dataset_normalized = []
for image, label in test_dataset:
    image_normalized = image  # No need to normalize the images
    label_normalized = normalize_test_data(label, x_min, x_max, y_min, y_max)
    test_dataset_normalized.append((image_normalized, label_normalized))

# Create the DataLoader for the normalized test dataset
test_loader = DataLoader(test_dataset_normalized, batch_size=batch_size, shuffle=False)

# Define the RNN model
class CursorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CursorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Get the last time step's output for each batch
        return out

# Define the hyperparameters and instantiate the RNN model
input_size = 48  # Number of features
hidden_size = 256  # Number of features in the hidden state
num_layers = 3  # Number of recurrent layers
output_size = 2  # Number of output dimensions (X and Y coordinates)
model = CursorRNN(input_size, hidden_size, num_layers, output_size).to(device)

# Choose loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# # Train the model
# num_epochs = 100
# losses = []  # To store the loss values
#
# for epoch in range(num_epochs):
#     epoch_loss = 0.0
#     for i in range(len(x_train)):
#         inputs = x_train[i].unsqueeze(0).to(device)  # Add batch dimension
#         targets = y_train[i].unsqueeze(0).to(device)  # Add batch dimension
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#
#     epoch_loss /= len(x_train)  # Calculate average loss for the epoch
#     losses.append(epoch_loss)  # Store the average loss
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')
#
# # Plot the loss
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Across Training')
# plt.show()
#
# #Save the model here
# torch.save(model.state_dict(), 'C:/Users/jeffr/PycharmProjects/EEG_RNN/model.pth')
#
# def eval(data_loader, net, criterion):
#     total_loss = 0.0
#     num_samples = 0
#
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             num_samples += len(inputs)
#
#     return total_loss / num_samples

# Load the pre-saved model state
model_path = "C:/Users/jeffr/PycharmProjects/EEG_RNN/model.pth" #This model has 81% accuracy
model.load_state_dict(torch.load(model_path))

# Testing the model
model.eval()  # Set the model to evaluation mode

test_loss = 0.0
with torch.no_grad():  # Disable gradient calculation during testing
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Normalize the labels before using them for the loss calculation
        normalized_labels = normalize_test_data(labels, x_min, x_max, y_min, y_max)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, normalized_labels)

        test_loss += loss.item()

    test_loss /= len(test_loader)

print(f'Test Loss: {test_loss:.4f}')

# Define the eval_plot function to compute forward pass and predictions
def eval_plot(data, net):
    with torch.no_grad():
        # Unpack the inputs and labels from the data loader
        inputs, labels = data

        # Move inputs and labels to the appropriate device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = net(inputs.unsqueeze(0))  # Add batch dimension
    return outputs[0].cpu().numpy(), labels.cpu().numpy()

# Select n consecutive images and plot the actual and predicted trajectories
n = 100
start = 15
selected_data = test_dataset_normalized[start:start + n]
predicted_labels = [eval_plot(data, model) for data in selected_data]

# Extract the predicted and actual labels
predicted_labels, actual_labels = zip(*predicted_labels)
diff = [np.linalg.norm(predicted_labels[i] - actual_labels[i]) for i in range(len(predicted_labels))]
print("Average difference between predicted and actual labels: ", np.mean(diff))

# Plot the predicted and actual trajectories
fig, ax = plt.subplots(dpi=300)
ms = 3
lw = 0.7

# Plot the predicted trajectory as a line
ax.plot([predicted_labels[i][0] for i in range(len(predicted_labels))], [predicted_labels[i][1] for i in range(len(predicted_labels))], label="predicted", color="red", lw=lw, marker='o', markersize=ms, markerfacecolor="none", markeredgewidth=0.7, linestyle='-', zorder=1)

# Plot the actual trajectory as a line
ax.plot([actual_labels[i][0] for i in range(len(actual_labels))], [actual_labels[i][1] for i in range(len(actual_labels))], label="actual", color="blue", lw=lw, marker='x', markersize=ms, linestyle='-', zorder=2)

# Plot the dots again to show the individual points
ax.plot([predicted_labels[i][0] for i in range(len(predicted_labels))], [predicted_labels[i][1] for i in range(len(predicted_labels))], color="red", lw=0, marker='o', markersize=ms, markerfacecolor="none", markeredgewidth=0.7, zorder=3)
ax.plot([actual_labels[i][0] for i in range(len(actual_labels))], [actual_labels[i][1] for i in range(len(actual_labels))], color="blue", lw=0, marker='x', markersize=ms, zorder=4)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Predicted vs Actual Trajectory")
ax.legend()
plt.show()

threshold = 0.1
num_correct = 0

# Calculate the test accuracy
for predicted, actual in zip(predicted_labels, actual_labels):
    distance = np.linalg.norm(predicted - actual)
    if distance <= threshold:
        num_correct += 1

test_accuracy = num_correct / len(predicted_labels) * 100.0
print(f"Test Accuracy: {test_accuracy:.2f}%")


threshold = 0.1
TP, FP, FN = 0, 0, 0

# Calculate TP, FP, and FN
for predicted, actual in zip(predicted_labels, actual_labels):
    distance = np.linalg.norm(predicted - actual)
    if distance <= threshold:
        TP += 1
    else:
        FP += 1
        FN += 1

# Calculate precision, recall, and F1 score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"F1 Score: {f1_score:.4f}")