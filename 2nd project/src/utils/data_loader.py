import sys
from pathlib import Path
import os

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from src.utils.cnn import load_data
from config.constants import Batch_size, Images_size

# Load data 
train_dir = os.path.join(root_dir, 'data', 'training')
valid_dir = os.path.join(root_dir, 'data', 'validation')

train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                    valid_dir, 
                                                    batch_size=Batch_size, 
                                                    img_size=Images_size) 

classnames = train_loader.dataset.classes