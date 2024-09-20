import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .autoencoder import DiagonalGaussianDistribution

# Residual block with projection
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

# ProjectLayer as per your provided structure
class ProjectLayer(nn.Module):
    def __init__(self, embedding_dim, proj_dim, drop_proj=0.3):
        super(ProjectLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, x):
        x = x.view(x.shape[0], self.embedding_dim)
        x = self.model(x)
        return x

# Encoder with larger kernel sizes and residual blocks
class Encoder(nn.Module):
    def __init__(self, input_channels=17, latent_dim=512):
        super(Encoder, self).__init__()
        # Convolution layers with larger kernel sizes
        self.conv1 = nn.Conv1d(input_channels, 24, kernel_size=16, stride=2, padding=3)  # (batch, 64, 125)
        self.conv2 = nn.Conv1d(24, 48, kernel_size=16, stride=2, padding=3)             # (batch, 128, 63)
        
        # ProjectLayer for embedding dimension projection
        self.proj_layer = ProjectLayer(48 * 56, 2*latent_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten and project using ProjectLayer
        x = x.view(x.size(0), -1)
        x = self.proj_layer(x)
        
        return x

# Decoder with residual blocks and larger kernel sizes
class Decoder(nn.Module):
    def __init__(self, output_channels=17, latent_dim=512):
        super(Decoder, self).__init__()
        # Fully connected layer to reshape latent vector
        self.fc = nn.Linear(latent_dim, 2*latent_dim)
        
        # ProjectLayer for embedding dimension projection
        self.proj_layer = ProjectLayer(2*latent_dim, 48 * 56)
        
        # Deconv layers with larger kernel sizes
        self.deconv1 = nn.ConvTranspose1d(48, 24, kernel_size=16, stride=2, padding=3)  # (batch, 256, 32)
        self.deconv2 = nn.ConvTranspose1d(24, output_channels, kernel_size=16, stride=2, padding=2)  # (batch, 128, 63)

        self.ln = nn.LayerNorm(252)
    def forward(self, z):
        # Fully connected layer and project using ProjectLayer
        x = F.relu(self.fc(z))
        x = self.proj_layer(x)
        x = x.view(x.size(0), 48, 56)
        
        # Deconv layers to reconstruct input
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)

        x = F.interpolate(x, size=(250), mode='linear', align_corners=False)
        return x

# VAE model that combines the encoder and decoder
class SpatialTemporalVAE(nn.Module):
    def __init__(self, num_ch=17, timesteps=250, latent_dim=512):
        super(SpatialTemporalVAE, self).__init__()
        self.encoder = Encoder(num_ch, latent_dim)
        self.decoder = Decoder(num_ch, latent_dim)


    def forward(self, x,sample_posterior=True, scale=1.0):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        recon = self.decoder(z)
        return posterior, z, recon
    def encode(self,eeg , scale=1.0):
        z = self.encoder(eeg)
        posterior = DiagonalGaussianDistribution(z, scale)
        
        return posterior
# Example usage
# vae = SpatialTemporalVAE(num_ch=17, latent_dim=512)

# # Example input (batch_size, channels, time_series_length)
# x = torch.randn(32, 17, 250)
# posterior, z, recon = vae(x)
# print(recon.shape)  # Should output: torch.Size([32, 17, 250])