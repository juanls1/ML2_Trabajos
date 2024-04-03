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

from config.constants import Model_used, extra_models, Criterion, Optimizer, Learning_rate, Number_epochs, Model_name
from src.utils.data_loader import num_classes, train_loader, valid_loader
from src.utils.cnn import CNN


# Carga y elección del modelo usado

# Pytorch has many pre-trained models that can be used for transfer learning
classification_models = torchvision.models.list_models(module=torchvision.models)

if Model_used not in classification_models.extend(extra_models):
    print(f"Model {Model_used} not found")
    print("Available models are:")
    print(classification_models.extend(extra_models))
    sys.exit()

# Change the model name according to the model used
if Model_used == 'resnet50':
    model_used = torchvision.models.resnet50(weights='DEFAULT')
else:
    raise ValueError(f"Model {Model_used} not supported")


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
    }
)


# Instancia el modelo de CNN

# Carga de parámetros

model = CNN(model_used, num_classes)

# Define el optimizador y la función de pérdida
if Criterion == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
else:
    raise ValueError(f'Criterion {Criterion} not supported')

if Optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
else:
    raise ValueError(f'Optimizer {Optimizer} not supported')

# Entrenamiento del modelo

for epoch in range(Number_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Zero gradient
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track training loss
        running_loss += loss.item()

        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate accuracy and loss
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # Log metrics to W&B
    wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})

# Guardado del modelo
model.save(Model_name)

# Finaliza el run de W&B
wandb.finish()