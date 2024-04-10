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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(model_used, num_classes)

model.to(device)

# Define el optimizador y la función de pérdida
criterion = nn.CrossEntropyLoss()
criterion.to(device)

if Optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
else:
    raise ValueError(f'Optimizer {Optimizer} not supported')

# Define un contador para el número máximo de épocas sin mejora en el entrenamiento
max_epochs_no_improve = 5  # Por ejemplo, puedes ajustar este valor según tus necesidades
epochs_no_improve = 0
best_train_loss = float('inf')  # Inicializa la mejor pérdida en la validación como infinito positivo
best_train_acc = 0.0  # Inicializa la mejor precisión en la validación como 0.0

# Entrenamiento del modelo
for epoch in range(Number_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()  # Poner el modelo en modo de entrenamiento
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        
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
            images, labels = images.to(device), labels.to(device)
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

    # Comprobar si hay mejora en la pérdida en la validación
    if train_loss < best_train_loss:

        best_train_loss = train_loss
        epochs_no_improve = 0  # Reinicia el contador de épocas sin mejora
    elif train_accuracy > best_train_acc:
        best_train_acc = train_accuracy
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

# Guardado del modelo
torch.save(model.state_dict(), Model_name)

# Finaliza el run de W&B
wandb.finish()

stop_time = time.time()

print(f"Training finished in {stop_time - start_time} seconds")
