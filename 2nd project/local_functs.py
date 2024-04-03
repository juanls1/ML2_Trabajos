import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# Visualize a few images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Función para mostrar una cuadrícula de imágenes con títulos
def show_images_grid(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0))  # Permute para cambiar las dimensiones (C, H, W) a (H, W, C)
        ax.set_title(titles[i], fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Define un Dataset personalizado para cargar la imagen
class CustomImageDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform

    def __len__(self):
        return 1  # Solo hay una imagen en la carpeta

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.image)
        return image, 0  # Establece la etiqueta en 0 para todas las imágenes