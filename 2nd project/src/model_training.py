import sys
from pathlib import Path
import torchvision
import torch
import torch.nn as nn
import wandb

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from config.variables import Model_used, extra_models, Criterion, Optimizer, Learning_rate, Number_epochs
from src.utils.data_loader import num_classes, train_loader, valid_loader
from src.utils.cnn import CNN
from src.utils.local_functs import CustomLoss

import time

start_time = time.time()

# Carga y elección del modelo usado

# Pytorch has many pre-trained models that can be used for transfer learning
classification_models = torchvision.models.list_models(module=torchvision.models)

classification_models.extend(extra_models)

if Model_used not in classification_models:

    print(f"Model {Model_used} not found")
    print("Available models are:")
    print(classification_models.extend(extra_models))
    sys.exit()

else:
    model_used = torchvision.models.__dict__[Model_used](weights='DEFAULT')

# Inicialización de wandb

# start a new wandb run to track this script
wandb.init(

    # set the wandb project where this run will be logged
    project="ML2-CNN-PROJECT",
    # track hyperparameters and run metadata
    config={
    "learning_rate": Learning_rate,
    "architecture": "my_trained_model",
    "dataset": "YourDataset",
    "epochs": Number_epochs,
    },
    dir=str(root_dir)  # o cualquier otra ruta que desees

)

# Instancia el modelo de CNN

# Carga de parámetros

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(model_used, num_classes)

model.to(device)

# Define el optimizador y la función de pérdida
if Criterion == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
    extra_criterion = None

elif Criterion == 'CustomLoss':
    criterion = nn.CrossEntropyLoss()
    extra_criterion = CustomLoss()
else:
    raise ValueError(f'Optimizer {Criterion} not supported')

criterion.to(device)

if Optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
else:
    raise ValueError(f'Optimizer {Optimizer} not supported')

# Registrar métricas en W&B
wandb.log({"BestModels_train_loss": 4, "BestModels_train_accuracy": 0, "BestModels_train_scores": 0,
               "BestModels_valid_loss": 4, "BestModels_valid_accuracy": 0, "BestModels_valid_scores": 0})


# Entrenamiento del modelo (cnn.py ha sido modificado para incluir el registro de métricas en W&B y otras métricas)
model.train_model(device, train_loader, valid_loader, 
                  optimizer, criterion, extra_criterion, Number_epochs)

# Finaliza el run de W&B
wandb.finish()

stop_time = time.time()

print(f"Training finished in {stop_time - start_time} seconds")