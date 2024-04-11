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

from config.variables import Model_used, extra_models, Criterion, Optimizer, Learning_rate, Number_epochs, Model_name, Max_iterations_change, threshold
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
if Criterion == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
else:
    raise ValueError(f'Optimizer {Criterion} not supported')

criterion.to(device)

if Optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
else:
    raise ValueError(f'Optimizer {Optimizer} not supported')

# Define un contador para el número máximo de épocas sin mejora en el entrenamiento
max_epochs_no_improve = Max_iterations_change  
epochs_no_improve = 0
best_train_loss = float('inf')  # Inicializa la mejor pérdida en la validación como infinito positivo
best_train_acc = 0.0  # Inicializa la mejor precisión en train como 0.0

# Registrar métricas en W&B
wandb.log({"BestModels_train_loss": 4, "BestModels_train_accuracy": 0, "BestModels_train_scores": 0,
               "BestModels_valid_loss": 4, "BestModels_valid_accuracy": 0, "BestModels_valid_scores": 0})


# Entrenamiento del modelo
for epoch in range(Number_epochs):

    running_loss = 0.0
    correct = 0
    total = 0
    score = 0

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

        # Get the probabilities and predicted classes
        top_probs, top_classes = torch.topk(outputs, k=2, dim=1)

        # Iterar sobre cada foto
        for i in range(outputs.shape[0]):
            # Calcular la diferencia de probabilidades para la foto actual
            diff = torch.abs(top_probs[i, 0] - top_probs[i, 1])
            
            # Comparar la diferencia con el umbral
            if diff < threshold:
                class1 = top_classes[i, 0]
                prob1 = top_probs[i, 0]
                class2 = top_classes[i, 1]
                prob2 = top_probs[i, 1]
                match_score = torch.where(class1 == labels[i], 0.8, torch.where(class2 == labels[i], 0.6, 0))
            else:
                class1 = top_classes[i, 0]
                prob1 = top_probs[i, 0]
                match_score = torch.where(class1 == labels[i], 1.0, 0)

            # Incrementar el contador de predicciones correctas
            correct += ((class1 == labels[i])).sum().item()
            # Sumar el score
            score += match_score

        # Incrementar el contador total
        total += labels.size(0)

    # Calculate accuracy and loss
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_scores = float(score) / total

    #------------------------------------------------------------

    # Calcular el accuracy y la pérdida en el conjunto de validación
    valid_running_loss = 0.0
    valid_correct = 0
    valid_total = 0
    valid_score = 0

    model.eval()  # Poner el modelo en modo de evaluación

    with torch.no_grad():

        for images, labels in valid_loader:

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            # Track accuracy

            # Get the probabilities and predicted classes
            top_probs, top_classes = torch.topk(outputs, k=2, dim=1)

            # Iterar sobre cada foto
            for i in range(outputs.shape[0]):
                # Calcular la diferencia de probabilidades para la foto actual
                diff = torch.abs(top_probs[i, 0] - top_probs[i, 1])
                
                # Comparar la diferencia con el umbral
                if diff < threshold:
                    class1 = top_classes[i, 0]
                    prob1 = top_probs[i, 0]
                    class2 = top_classes[i, 1]
                    prob2 = top_probs[i, 1]
                    match_score = torch.where(class1 == labels[i], 0.8, torch.where(class2 == labels[i], 0.6, 0))
                else:
                    class1 = top_classes[i, 0]
                    prob1 = top_probs[i, 0]
                    match_score = torch.where(class1 == labels[i], 1.0, 0)

                # Incrementar el contador de predicciones correctas
                valid_correct += ((class1 == labels[i])).sum().item()
                # Sumar el score
                valid_score += match_score

            # Incrementar el contador total
            valid_total += labels.size(0)

    # Calcular accuracy y loss en el conjunto de validación
    valid_loss = valid_running_loss / len(valid_loader)
    valid_accuracy = valid_correct / valid_total
    valid_scores = valid_score / valid_total

    # Registrar métricas en W&B
    wandb.log({"BestModels_train_loss": train_loss, "BestModels_train_accuracy": train_accuracy, "BestModels_train_scores": train_scores,
               "BestModels_valid_loss": valid_loss, "BestModels_valid_accuracy": valid_accuracy, "BestModels_valid_scores": valid_scores})

    # Comprobar si hay mejora en la pérdida en la validación
    if train_loss < best_train_loss:

        best_train_loss = train_loss
        epochs_no_improve = 0  # Reinicia el contador de épocas sin mejora

    elif train_accuracy > best_train_acc:

        best_train_acc = train_accuracy
        epochs_no_improve = 0

    else:
        epochs_no_improve += 1

    if epochs_no_improve > max_epochs_no_improve:
        break


# Guardado del modelo
torch.save(model.state_dict(), Model_name)

# Finaliza el run de W&B
wandb.finish()

stop_time = time.time()

print(f"Training finished in {stop_time - start_time} seconds")