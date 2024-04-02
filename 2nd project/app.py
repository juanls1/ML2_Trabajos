import streamlit as st
from PIL import Image
import torch
from cnn import CNN
import torchvision
from data_loader import num_classes
from cnn import load_model_weights

def load_model(model_path, classes):
    model_weights = load_model_weights(model_path)
    model = CNN(torchvision.models.resnet50(weights='DEFAULT'), classes)
    model.load_state_dict(model_weights)
    return model

def main():
    st.title("Detección de objetos")
    st.markdown("¡Bienvenidos a la aplicación web de detección de objetos del grupo Lechuga!")
    
    # Widget para cargar una imagen
    image_file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
    
    # Selección del modelo
    model_option = st.selectbox("Selecciona el modelo a utilizar", ["Modelo 1 (1 epoch)", "Modelo 2 (4 epochs)"])
    
    if model_option == "Modelo 1 (1 epoch)":
        model_path = 'C:/Users/juanl/Documents/Master/2o cuatri/ML2/Codes/Entregas/ML2_Trabajos/2nd project/models/resnet50-1epoch'
    elif model_option == "Modelo 2 (4 epochs)":
        model_path = 'C:/Users/juanl/Documents/Master/2o cuatri/ML2/Codes/Entregas/ML2_Trabajos/2nd project/models/resnet50-1epoch'

    used_classes = num_classes
    
    # Cargar el modelo

    model = load_model(model_path, used_classes)
    
    if image_file is not None:
        # Preprocesar la imagen
        image = Image.open(image_file)
        image = image.convert('L')  # Convertir la imagen a blanco y negro
        image = image.resize((224, 224))  # ResNet50 requires 224x224 images
        
        # Convertir la imagen a un tensor de PyTorch
        transform = torchvision.transforms.ToTensor()
        image = transform(image)
        
        # Normalizar la imagen
        normalize = torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize(image)
        
        # Añadir dimensión batch
        image = image.unsqueeze(0)
        
        # Realizar la predicción
        predicted_labels = model.predict(image)

        #predicted_label = torch.argmax(predicted_labels, dim=1).item()

        # Mostrar la imagen y la predicción
        st.image(image_file, caption='Imagen cargada', use_column_width=True)
        st.write(f'Clase predicha: {predicted_labels}')

if __name__ == "__main__":
    main()