import numpy as np
import matplotlib.pyplot as plt
import os

def generate_vortex_field(size=64, num_vortices=5):
	x = np.linspace(-1, 1, size)
	y = np.linspace(-1, 1, size)

	X, Y = np.meshgrid(x, y)

	field = np.zeros((size, size))

	for _ in range(num_vortices):
		x0, y0 = np.random.uniform(-0.8, 0.8, 2)
		strength = np.random.uniform(-1.0, 1.0)
		sigma = np.random.uniform(0.05, 0.2)

		gauss = strength * np.exp(-((X -x0)**2 + (Y - y0)**2) / (2 * sigma**2))
		field += gauss

	return field


def create_dataset(num_samples=1000, save_path="data/"):
	os.makedirs(save_path, exist_ok=True)	
	for i in range(num_samples):
		field = generate_vortex_field()
		np.save(f"{save_path}/vortex_{i:04d}.npy", field)


if __name__ == "__main__":
	create_dataset(num_samples=5000)