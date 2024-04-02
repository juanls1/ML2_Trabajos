import numpy as np
import matplotlib.pyplot as plt
import torch

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



def preprocess_image(data_loader):
    """Preprocess images from the given data loader.

    Args:
        data_loader: DataLoader with the images to preprocess.

    Returns:
        preprocessed_images: List of preprocessed images.
    """
    preprocessed_images = []
    for images, _ in data_loader:
        # Resize image to 224x224 (required by ResNet50)
        resized_images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        # Convert images to numpy arrays
        image_arrays = resized_images.numpy()
        preprocessed_images.extend(image_arrays)
    return preprocessed_images