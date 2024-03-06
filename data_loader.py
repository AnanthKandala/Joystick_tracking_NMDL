import torch
from torch.utils.data import Dataset, DataLoader, random_split
from data_cleaning import generate_clean_data


class CustomDataset(Dataset):
    def __init__(self, snipped_data):
        super(CustomDataset, self).__init__()
        # Load or initialize your data
        self.data = [torch.tensor(snip[0], dtype=torch.float) for snip in snipped_data]
        self.labels = [torch.tensor(snip[1], dtype=torch.float) for snip in snipped_data] # Only for supervised tasks
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]  # Only for supervised tasks
        return sample, label
    
def generate_data_loader(data_file):
    cleaned_data = generate_clean_data(data_file)
    dataset = CustomDataset(cleaned_data)
    per_train = 0.8 #percent of data to use for training
    train_size = int(per_train * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    batch_size = 32 #create batches
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return cleaned_data, train_loader, test_loader
