import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob

class VortexDataset(Dataset):
	def __init__(self, path):
		self.files = glob(os.path.join(path, "*.npy"))


	def __len__(self):
		return len(self.files)


	def __getitem__(self, idx):
		data = np.load(self.files[idx]).astype(np.float32)
		data = (data - data.min()) / (data.max() - data.min() + 1e-8) # Normalize [0, 1]

		return torch.tensor(data).unsqueeze(0)



class VAE(nn.Module):
	def __init__(self, latent_dim=8):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(),
			nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
			nn.Flatten()
		)

		self.fc_mu = nn.Linear(32 * 16 * 16, latent_dim)
		self.fc_logvar = nn.Linear(32 * 16 * 16, latent_dim)
		self.fc_dec = nn.Linear(latent_dim, 32 * 16 * 16)
		self.decoder = nn.Sequential(
			nn.Unflatten(1, (32, 16, 16)),
			nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
			nn.ConvTranspose2d(16, 1, 4, 2, 1),  nn.Sigmoid()
		)


	def encode(self, x):
		h = self.encoder(x)
		return self.fc_mu(h), self.fc_logvar(h)


	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z):
		h = self.fc_dec(z)
		return self.decoder(h)


	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z)
		return recon, mu, logvar



def loss_fn(recon, x, mu, logvar):
	beta = 0.001
	recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
	kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return recon_loss + beta * kl



def train():
	dataset = VortexDataset("data")
	loader = DataLoader(dataset, batch_size=32, shuffle=True)
	model = VAE(latent_dim=32)
	optimizer = optim.Adam(model.parameters(), lr=1e-4)


	model.train()
	for epoch in range(50):
		total_loss = 0
		for batch in loader:
			optimizer.zero_grad()
			recon, mu, logvar = model(batch)
			loss = loss_fn(recon, batch, mu, logvar)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

		print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

	torch.save(model.state_dict(), "vae_vortex.pt")



if __name__ == "__main__":
	train()













