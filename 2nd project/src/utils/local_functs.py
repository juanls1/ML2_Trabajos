import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from src.utils.cnn import load_model_weights
from src.utils.cnn import CNN
from config.constants import Figsize, Fontsize

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
    fig, axes = plt.subplots(rows, cols, figsize=Figsize)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0))  # Permute para cambiar las dimensiones (C, H, W) a (H, W, C)
        ax.set_title(titles[i], fontsize=Fontsize)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def load_model(model_path, classes):
    model_weights = load_model_weights(model_path)
    model = CNN(torchvision.models.resnet50(weights='DEFAULT'), classes)
    model.load_state_dict(model_weights)
    return model


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