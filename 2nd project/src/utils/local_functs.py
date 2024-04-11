import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import torch.nn as nn

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from config.variables import Figsize, Fontsize, threshold

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


class CustomLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(CustomLoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        # Calcular las probabilidades utilizando softmax
        probabilities = torch.softmax(input, dim=1)

        # Obtener las dos clases más probables para cada instancia
        top_probs, top_classes = torch.topk(probabilities, k=2, dim=1)

        # Inicializar la puntuación
        score = 0.0

        # Iterar sobre cada instancia
        for i in range(input.shape[0]):
            # Calcular la diferencia de probabilidades
            diff = torch.abs(top_probs[i, 0] - top_probs[i, 1])

            # Comparar la diferencia con el umbral
            if diff < self.threshold:
                # Obtener las clases predichas
                class1 = top_classes[i, 0]
                class2 = top_classes[i, 1]
                # Calcular la puntuación según la lógica dada
                score += torch.where(class1 == target[i], 0.8, torch.where(class2 == target[i], 0.6, 0))
            else:
                # Obtener la clase predicha
                class1 = top_classes[i, 0]
                # Calcular la puntuación según la lógica dada
                score += torch.where(class1 == target[i], 1.0, 0)

        # Calcular el promedio de la puntuación
        loss = -torch.mean(score)

        return loss
    