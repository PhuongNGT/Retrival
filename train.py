import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import ConvAutoencoder_v2
from barbar import Bar
import time, copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Dataset ===
class OxfordDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# === Huấn luyện ===
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    all_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0

            for inputs in Bar(dataloaders[phase]):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f}")
            all_losses.append(epoch_loss)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    plot_loss(all_losses)
    return model

# === Vẽ loss ===
def plot_loss(losses):
    train_loss = losses[::2]
    val_loss = losses[1::2]
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('loss_plot.png')
    print("📉 Saved loss plot: loss_plot.png")

# === Main ===
if __name__ == '__main__':
    dataset_dir = Path("dataset/oxbuild_images")
    image_paths = sorted([str(p) for p in dataset_dir.glob("*.jpg")])

    # Dataset split
    train_val, test = train_test_split(image_paths, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15, random_state=42)

    train_set = OxfordDataset(train)
    val_set = OxfordDataset(val)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=32, shuffle=True),
        'val': DataLoader(val_set, batch_size=32)
    }
    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}

    model = ConvAutoencoder_v2().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer)

    torch.save(model.state_dict(), "conv_autoencoderv2_oxford.pt")
    print("✅ Saved model: conv_autoencoderv2_oxford.pt")

