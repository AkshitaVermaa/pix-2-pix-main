import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import time


# Dataset class to load paired images from multiple subject folders
class PairedImageDataset(Dataset):
    def __init__(self, root_dir, image_size=256, mode='train'):
        """
        Args:
            root_dir: Path to dataset folder containing subject-1, subject-2, etc.
            image_size: Size to resize images to
            mode: 'train' or 'test'
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        
        # Find all subject folders
        self.image_pairs = []
        subject_folders = sorted(glob.glob(os.path.join(root_dir, 'subject-*')))
        
        for subject_folder in subject_folders:
            side_path = os.path.join(subject_folder, 'side.jpg')
            front_path = os.path.join(subject_folder, 'front.jpg')
            
            if os.path.exists(side_path) and os.path.exists(front_path):
                self.image_pairs.append((side_path, front_path))
        
        print(f"Found {len(self.image_pairs)} image pairs")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        side_path, front_path = self.image_pairs[idx]
        
        # Load images
        side_img = Image.open(side_path).convert('RGB')
        front_img = Image.open(front_path).convert('RGB')
        
        # Apply transforms
        side_tensor = self.transform(side_img)
        front_tensor = self.transform(front_img)
        
        return side_tensor, front_tensor


# Generator Network (U-Net architecture)
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64, normalize=False)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512)
        self.enc6 = self.conv_block(512, 512)
        self.enc7 = self.conv_block(512, 512)
        self.enc8 = self.conv_block(512, 512, normalize=False)
        
        # Decoder (upsampling)
        self.dec1 = self.deconv_block(512, 512, dropout=True)
        self.dec2 = self.deconv_block(1024, 512, dropout=True)
        self.dec3 = self.deconv_block(1024, 512, dropout=True)
        self.dec4 = self.deconv_block(1024, 512)
        self.dec5 = self.deconv_block(1024, 256)
        self.dec6 = self.deconv_block(512, 128)
        self.dec7 = self.deconv_block(256, 64)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def conv_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def deconv_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder with skip connections (U-Net)
        d1 = self.dec1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        
        output = self.final(d7)
        return output


# Discriminator Network (PatchGAN)
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6):  # 3 for input + 3 for target
        super(PatchGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: 6 x 256 x 256
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 128 x 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 64 x 64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 x 32 x 32
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 x 31 x 31
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # Output: 1 x 30 x 30 (PatchGAN)
        )
    
    def forward(self, x, y):
        # Concatenate input and target
        combined = torch.cat([x, y], dim=1)
        return self.model(combined)


# Pix2Pix cGAN Model
class Pix2PixModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', lr=0.0002):
        self.device = device
        
        # Initialize networks
        self.generator = UNetGenerator().to(device)
        self.discriminator = PatchGANDiscriminator().to(device)
        
        # Initialize weights
        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)
        
        # Loss functions
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        self.criterion_L2 = nn.MSELoss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Loss weights
        self.lambda_L1 = 100
        self.lambda_L2 = 100
        
        print(f"Model initialized on {device}")
    
    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def train_step(self, side_imgs, front_imgs):
        side_imgs = side_imgs.to(self.device)
        front_imgs = front_imgs.to(self.device)
        
        batch_size = side_imgs.size(0)
        
        # Real and fake labels
        real_label = torch.ones(batch_size, 1, 30, 30).to(self.device)
        fake_label = torch.zeros(batch_size, 1, 30, 30).to(self.device)
        
        # -------------------
        # Train Discriminator
        # -------------------
        self.optimizer_D.zero_grad()
        
        # Generate fake images
        fake_front = self.generator(side_imgs)
        
        # Real loss
        pred_real = self.discriminator(side_imgs, front_imgs)
        loss_D_real = self.criterion_GAN(pred_real, real_label)
        
        # Fake loss
        pred_fake = self.discriminator(side_imgs, fake_front.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, fake_label)
        
        # Total discriminator loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.optimizer_D.step()
        
        # ---------------
        # Train Generator
        # ---------------
        self.optimizer_G.zero_grad()
        
        # GAN loss (Adversarial / BCE)
        pred_fake = self.discriminator(side_imgs, fake_front)
        loss_G_GAN = self.criterion_GAN(pred_fake, real_label)
        
        # L1 loss
        loss_G_L1 = self.criterion_L1(fake_front, front_imgs) * self.lambda_L1
        
        # L2 loss
        loss_G_L2 = self.criterion_L2(fake_front, front_imgs) * self.lambda_L2
        
        # Total generator loss
        loss_G = loss_G_GAN + loss_G_L1 + loss_G_L2
        loss_G.backward()
        self.optimizer_G.step()
        
        return {
            'loss_D': loss_D.item(),
            'loss_G': loss_G.item(),
            'loss_G_GAN': loss_G_GAN.item(),
            'loss_G_L1': loss_G_L1.item(),
            'loss_G_L2': loss_G_L2.item()
        }
    
    def generate(self, side_img):
        self.generator.eval()
        with torch.no_grad():
            side_img = side_img.to(self.device)
            fake_front = self.generator(side_img)
        self.generator.train()
        return fake_front
    
    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }, path)
        print(f"Checkpoint saved at {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']


# Training function
def train(model, dataloader, num_epochs=200, save_interval=10):
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    history = {'loss_D': [], 'loss_G': [], 'loss_G_GAN': [], 'loss_G_L1': [], 'loss_G_L2': []}
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses = {'loss_D': 0, 'loss_G': 0, 'loss_G_GAN': 0, 'loss_G_L1': 0, 'loss_G_L2': 0}
        
        for i, (side_imgs, front_imgs) in enumerate(dataloader):
            losses = model.train_step(side_imgs, front_imgs)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
            history[key].append(epoch_losses[key])
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"L1: {epoch_losses['loss_G_L1']:.4f} "
              f"L2: {epoch_losses['loss_G_L2']:.4f} "
              f"BCE: {epoch_losses['loss_G_GAN']:.4f} "
              f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            model.save_checkpoint(f'checkpoints/pix2pix_epoch_{epoch+1}.pth', epoch+1)
            
            # Generate sample images
            model.generator.eval()
            with torch.no_grad():
                side_imgs, front_imgs = next(iter(dataloader))
                side_imgs = side_imgs.to(model.device)
                fake_front = model.generator(side_imgs)
                
                # Save sample results
                save_sample_images(side_imgs, front_imgs, fake_front, epoch+1)
            model.generator.train()
    
    # Plot training curves
    plot_training_curves(history)
    
    return history


def save_sample_images(side_imgs, front_imgs, fake_front, epoch):
    """Save sample images for visualization"""
    def denormalize(img):
        img = img * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
        return img.cpu().numpy().transpose(1, 2, 0)
    
    fig, axes = plt.subplots(min(4, side_imgs.size(0)), 3, figsize=(12, 4 * min(4, side_imgs.size(0))))
    if side_imgs.size(0) == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(4, side_imgs.size(0))):
        # Side profile
        axes[i, 0].imshow(denormalize(side_imgs[i]))
        axes[i, 0].set_title('Side Profile')
        axes[i, 0].axis('off')
        
        # Real front
        axes[i, 1].imshow(denormalize(front_imgs[i]))
        axes[i, 1].set_title('Real Front')
        axes[i, 1].axis('off')
        
        # Generated front
        axes[i, 2].imshow(denormalize(fake_front[i]))
        axes[i, 2].set_title('Generated Front')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results/sample_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history):
    """Plot training loss curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(history['loss_D'])
    axes[0, 0].set_title('Discriminator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['loss_G'])
    axes[0, 1].set_title('Generator Total Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(history['loss_G_GAN'])
    axes[0, 2].set_title('Generator BCE (Adversarial) Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(history['loss_G_L1'])
    axes[1, 0].set_title('Generator L1 Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['loss_G_L2'])
    axes[1, 1].set_title('Generator L2 Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    axes[1, 2].axis('off')  # Empty subplot
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


# Main execution
if __name__ == '__main__':
    # Configuration
    DATASET_PATH = 'dataset'
    IMAGE_SIZE = 256
    BATCH_SIZE = 4
    NUM_EPOCHS = 240
    LEARNING_RATE = 0.0002
    
    # Create dataset and dataloader
    dataset = PairedImageDataset(DATASET_PATH, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Initialize model
    model = Pix2PixModel(lr=LEARNING_RATE)
    
    # Train the model
    print("\nStarting training...")
    history = train(model, dataloader, num_epochs=NUM_EPOCHS, save_interval=10)
    
    # Save final model
    model.save_checkpoint('checkpoints/pix2pix_final.pth', NUM_EPOCHS)
    print("\nTraining completed!")
