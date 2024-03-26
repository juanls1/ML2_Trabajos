import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar una cuadrícula de imágenes con títulos
def show_images_grid(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0))  # Permute para cambiar las dimensiones (C, H, W) a (H, W, C)
        ax.set_title(titles[i], fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()