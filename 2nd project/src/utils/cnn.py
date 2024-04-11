import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tempfile import TemporaryDirectory
import wandb
import numpy as np

# Constants necessary for the project
import sys
from pathlib import Path

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from config.variables import Unfreezed_layers, threshold, Saved_epochs

class CNN(nn.Module):
    """Convolutional Neural Network model for image classification."""
    
    def __init__(self, base_model, num_classes, unfreezed_layers=Unfreezed_layers):
        """CNN model initializer.

        Args:
            base_model: Pre-trained model to use as the base.
            num_classes: Number of classes in the dataset.
            unfreezed_layers: Number of layers to unfreeze from the base model.

        """
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Freeze convolutional layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze specified number of layers
        if unfreezed_layers > 0:
            for layer in list(self.base_model.children())[-unfreezed_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True


        try:
            # Add a new softmax output layer
            self.fc = nn.Sequential(
                nn.Linear(self.base_model.classifier[-1].in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, num_classes),
                nn.Softmax(dim=1)
            )

            # Try replacing the last layer of the base model for vgg
            self.base_model.classifier[-1] = nn.Identity()
        except AttributeError:
            try:
                # Add a new softmax output layer
                self.fc = nn.Sequential(
                    nn.Linear(self.base_model.fc.in_features, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(1024, num_classes),
                    nn.Softmax(dim=1)
                )

                # If that fails, try replacing the last layer of the base model for resnet
                self.base_model.fc = nn.Identity()
            except AttributeError:
                # If neither works, raise an error
                raise AttributeError("Neither 'classifier' nor 'fc' attribute found in the base model.")

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input data.
        """
        x = self.base_model(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_model(self, 
                    device,
                    train_loader, 
                    valid_loader, 
                    optimizer, 
                    criterion, 
                    extra_criterion,
                    epochs, 
                    nepochs_to_save=Saved_epochs):
        """Train the model and save the best one based on validation accuracy.
        
        Args:
            train_loader: DataLoader with training data.
            valid_loader: DataLoader with validation data.
            optimizer: Optimizer to use during training.
            criterion: Loss function to use during training.
            epochs: Number of epochs to train the model.
            nepochs_to_save: Number of epochs to wait before saving the model.

        Returns:
            history: A dictionary with the training history.
        """
        with TemporaryDirectory() as temp_dir:
            best_model_path = os.path.join(temp_dir, 'best_model.pt')
            best_accuracy = 0.0
            best_score = 0.0
            torch.save(self.state_dict(), best_model_path)

            history = {'train_loss': [], 'train_accuracy': [], 'train_score': [], 
                       'valid_loss': [], 'valid_accuracy': [], 'valid_score': []}


            # Entrenamiento del modelo
            for epoch in range(epochs):

                train_loss = 0.0
                train_accuracy = 0.0
                train_score = 0.0

                self.train()  # Poner el modelo en modo de entrenamiento

                for images, labels in train_loader:

                    # Zero gradient
                    optimizer.zero_grad()
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    outputs = self(images)
                    loss = criterion(outputs, labels)

                    if extra_criterion is not None:
                        loss2 = extra_criterion(outputs, labels)
                        loss.data = torch.tensor(3*np.exp(float(loss2.data)/len(train_loader.dataset)))

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    # Track training loss
                    train_loss += loss.item()

                    # Track accuracy (equivalent to train_accuracy += (outputs.argmax(1) == labels).sum().item())

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
                            train_score += float(torch.where(class1 == labels[i], 0.8, torch.where(class2 == labels[i], 0.6, 0)))
                        else:
                            class1 = top_classes[i, 0]
                            train_score += float(torch.where(class1 == labels[i], 1.0, 0))

                        # Incrementar el contador de predicciones correctas
                        train_accuracy += ((class1 == labels[i])).sum().item()


                # Calculate accuracy and loss
                train_loss /= len(train_loader)
                train_accuracy /= len(train_loader.dataset)
                train_score /= len(train_loader.dataset)
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_accuracy)
                history['train_score'].append(train_score)

                print(f'Epoch {epoch + 1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Train Accuracy: {train_accuracy:.4f}, '
                      f'Train Score: {train_score:.4f} ')

                #------------------------------------------------------------

                # Calcular el accuracy y la pérdida en el conjunto de validación
                valid_loss = 0.0
                valid_accuracy = 0
                valid_score = 0

                self.eval()  # Poner el modelo en modo de evaluación

                with torch.no_grad():

                    for images, labels in valid_loader:

                        images, labels = images.to(device), labels.to(device)
                        outputs = self(images)
                        loss = criterion(outputs, labels)
                        valid_loss += loss.item()

                        if extra_criterion is not None:
                            loss2 = extra_criterion(outputs, labels)
                            loss.data = torch.tensor(3*np.exp(float(loss2.data)/len(valid_loader.dataset)))

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
                                valid_score += torch.where(class1 == labels[i], 0.8, torch.where(class2 == labels[i], 0.6, 0))
                            else:
                                class1 = top_classes[i, 0]
                                valid_score += torch.where(class1 == labels[i], 1.0, 0)

                            # Incrementar el contador de predicciones correctas
                            valid_accuracy += ((class1 == labels[i])).sum().item()


                # Calcular accuracy y loss en el conjunto de validación
                valid_loss /= len(valid_loader)
                valid_accuracy /= len(valid_loader.dataset)
                valid_score /= len(valid_loader.dataset)
                history['valid_loss'].append(valid_loss)
                history['valid_accuracy'].append(valid_accuracy)
                history['valid_score'].append(valid_score)

                print(f'Epoch {epoch + 1}/{epochs} - '
                        f'Validation Loss: {valid_loss:.4f}, '
                        f'Validation Accuracy: {valid_accuracy:.4f}, '
                        f'Validation Score: {valid_score:.4f} ')
                
                if epoch % nepochs_to_save == 0:
                    if valid_accuracy > best_accuracy or valid_score > best_score:
                        best_accuracy = valid_accuracy
                        best_score = valid_score
                        torch.save(self.state_dict(), best_model_path)

                # Registrar métricas en W&B
                wandb.log({"BestModels_train_loss": train_loss, "BestModels_train_accuracy": train_accuracy, "BestModels_train_scores": train_score,
                        "BestModels_valid_loss": valid_loss, "BestModels_valid_accuracy": valid_accuracy, "BestModels_valid_scores": valid_score})

                
            torch.save(self.state_dict(), best_model_path)    
            self.load_state_dict(torch.load(best_model_path))
            return history
        
    def predict(self, data_loader):
        """Predict the classes of the images in the data loader.

        Args:
            data_loader: DataLoader with the images to predict.

        Returns:
            predicted_labels: Predicted classes.
        """
        self.eval()
        predicted_labels = []
        for images, _ in data_loader:
            outputs = self(images)
            predicted_labels.extend(outputs.argmax(1).tolist())
        return predicted_labels
    
    def save(self, filename: str):
        """Save the model to disk.

        Args:
            filename: Name of the file to save the model.
        """
        # If the directory does not exist, create it
        os.makedirs(os.path.dirname('2nd project/models/'), exist_ok=True)

        # Full path to the model
        filename = os.path.join('2nd project/models', filename)

        # Save the model
        torch.save(self.state_dict(), filename+'.pt')

    @staticmethod
    def _plot_training(history):
        """Plot the training history.

        Args:
            history: A dictionary with the training history.
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['valid_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


def load_data(train_dir, valid_dir, test_dir, batch_size, img_size):
    """Load and transform the training and validation datasets.

    Args:
        train_dir: Path to the training dataset.
        valid_dir: Path to the validation dataset.
        batch_size: Number of images per batch.
        img_size: Expected size of the images.

    Returns:
        train_loader: DataLoader with the training dataset.
        valid_loader: DataLoader with the validation dataset.
        num_classes: Number of classes in the dataset.
    """
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30), # Rotate the image by a random angle
        transforms.RandomResizedCrop(img_size), # Crop the image to a random size and aspect ratio
        transforms.RandomHorizontalFlip(), # Horizontally flip the image with a 50% probability
        transforms.ToTensor() 
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor() 
    ])

    test_transforms = valid_transforms

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, len(train_data.classes)

def load_model_weights(filename: str):
        """Load a model from disk.
        IMPORTANT: The model must be initialized before loading the weights.
        Args:
            filename: Name of the file to load the model.
        """
        # Full path to the model
        filename = os.path.join('models', filename)

        # Load the model
        state_dict = torch.load(filename+'.pt')
        return state_dict