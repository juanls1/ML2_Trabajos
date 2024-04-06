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

from config.variables import Model_used, extra_models, Criterion, Optimizer, Learning_rate, Number_epochs, Model_name, Max_iterations_change
from src.utils.data_loader import num_classes, train_loader, valid_loader
from src.utils.cnn import CNN

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

# Define un contador para el número máximo de iteraciones sin cambio en error y precisión
max_iterations_without_change = Max_iterations_change
iterations_without_change = 0

# Entrenamiento del modelo
for epoch in range(Number_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    # Entrenamiento
    model.train()  # Poner el modelo en modo de entrenamiento
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

    # Calcular el accuracy y la pérdida en el conjunto de validación
    valid_running_loss = 0.0
    valid_correct = 0
    valid_total = 0

    model.eval()  # Poner el modelo en modo de evaluación

    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    # Calcular accuracy y loss en el conjunto de validación
    valid_loss = valid_running_loss / len(valid_loader)
    valid_accuracy = valid_correct / valid_total

    # Registrar métricas en W&B
    wandb.log({"valid_loss": valid_loss, "valid_accuracy": valid_accuracy})

    # Verifica si el error y la precisión no cambian
    if epoch > 0:
        if train_loss == prev_loss and train_accuracy == prev_accuracy:
            iterations_without_change += 1
        else:
            iterations_without_change = 0

        # Si no hay cambio durante un número específico de iteraciones, termina la ejecución
        if iterations_without_change >= max_iterations_without_change:
            print("El error y la precisión no han cambiado en las últimas iteraciones. Terminando la ejecución.")
            break

    # Guarda el error y la precisión de esta iteración para compararlos en la siguiente
    prev_loss = train_loss
    prev_accuracy = train_accuracy

# Guardado del modelo
model.save(Model_name)

# Finaliza el run de W&B
wandb.finish()

stop_time = time.time()

print(f"Training finished in {stop_time - start_time} seconds")
