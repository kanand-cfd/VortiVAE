import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

LATENT_DIM = 32 # 8, 16, 32

class VAE(nn.Module):
	def __init__(self, latent_dim=LATENT_DIM):
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
			nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid()
		)

	def decode(self, z):
		h = self.fc_dec(z)
		return self.decoder(h)



@st.cache_resource
def load_model():
	model = VAE()
	model.load_state_dict(torch.load("vae_vortex.pt", map_location="cpu"))
	model.eval()
	return model


# def show_image(img_tensor):
#	img = img_tensor.squeeze().detach().numpy()
#	fig, ax = plt.subplots()
#	ax.imshow(img, cmap="viridis")
#	ax.axis("off")
#	st.pyplot(fig)

def show_image(img_tensor, zoom=4):
    img = img_tensor.squeeze().detach().numpy()

    fig, ax = plt.subplots(figsize=(zoom, zoom))  # Smaller zoom = crisper
    ax.imshow(img, cmap="viridis", interpolation="nearest")  
    ax.axis("off")
    st.pyplot(fig)



st.title(" Latent Vortex Explorer ")

model = load_model()

z = torch.zeros((1, LATENT_DIM))
cols = st.columns(LATENT_DIM)

st.sidebar.header("Latent Vector")
for i in range(LATENT_DIM):
    z[0, i] = st.sidebar.slider(f"z[{i}]", -3.0, 3.0, 0.0, 0.1, key=f"z{i}")


with torch.no_grad():
	out = model.decode(z)


st.subheader(" Generated Vorticity Image ")

zoom = st.slider("Image Zoom", 1, 10, 4)
show_image(out[0], zoom=zoom)

