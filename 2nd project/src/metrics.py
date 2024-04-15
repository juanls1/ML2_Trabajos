import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from utils.data_loader import num_classes, train_loader, valid_loader, test_loader
from utils.cnn import CNN
from utils.data_loader import num_classes, classnames
from utils.cnn import load_model_weights
from config.variables import threshold
import os
import sys
from pathlib import Path

used_classes = num_classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_loss = 0.0
test_accuracy = 0
test_score = 0

model_used = torchvision.models.resnext101_32x8d(weights='DEFAULT')

model = CNN(model_used, used_classes)


root_dir = Path(__file__).resolve().parent.parent

sys.path.append(str(root_dir))
Models_dir = os.path.join(root_dir, 'models')

model_path = os.path.join(Models_dir, 'resnext101_32x8d-LR_0.0001-NE_10-UL_8-C_CustomLoss-O_Adam')

model_weights = load_model_weights(model_path, map_location=device)

model.load_state_dict(model_weights)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
model.to(device)

model.eval()

with torch.no_grad():

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

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
                class2 = top_classes[i, 1]
                test_score += torch.where(class1 == labels[i], 0.8, torch.where(class2 == labels[i], 0.6, 0))
            else:
                class1 = top_classes[i, 0]
                test_score += torch.where(class1 == labels[i], 1.0, 0)

            # Incrementar el contador de predicciones correctas
            test_accuracy += ((class1 == labels[i])).sum().item()


# Calcular accuracy y loss en el conjunto de testación
test_loss /= len(test_loader)
test_accuracy /= len(test_loader.dataset)
test_score /= len(test_loader.dataset)

print(f'testation Loss: {test_loss:.4f}, '
        f'testation Accuracy: {test_accuracy:.4f}, '
        f'testation Score: {test_score:.4f} ')