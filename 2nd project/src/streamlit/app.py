import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))

from src.utils.data_loader import num_classes, classnames
from src.utils.cnn import load_model_weights, CNN
from src.utils.local_functs import CustomImageDataset
from config.constants import Images_size, Images_types, Disp_Models, Models_paths


def main():
    st.set_page_config(page_title="ML2 - CNN", layout="centered")
    st.title("Clasificación de imágenes con CNNs")
    st.markdown("¡Bienvenidos a la aplicación web Canonist.ia de clasificación de imágenes del grupo compuesto por Alberto, Jorge, Nacho y Juan!")
    
    # Widget para cargar una imagen
    image_file = st.file_uploader("### Cargar imagen", type=Images_types)
    
    # Selección del modelo
    model_option = st.selectbox("### Selecciona el modelo a utilizar", Disp_Models)

    # Seleccionar el modelo a utilizar
    model_path = Models_paths[Disp_Models.index(model_option)]

    used_classes = num_classes
    
    # Cargar el modelo

    model_weights = load_model_weights(model_path)

    # Change the model name according to the model used

    if model_path.split('\\')[-1].split('-')[0] == 'resnet50':
        model_used = torchvision.models.resnet50(weights='DEFAULT')
    else:
        raise ValueError(f"#### Model {Models_paths} not supported")
    
    model = CNN(model_used, used_classes)
    
    model.load_state_dict(model_weights)
    
    if image_file is not None:
        # Preprocesar la imagen
        image = Image.open(image_file)

        img_size = Images_size

        streamlit_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=3),  # Convertir a RGB si es necesario
                transforms.ToTensor() 
            ])


        # Crea una instancia del Dataset personalizado
        streamlit_data = CustomImageDataset(image, transform=streamlit_transforms)

        # Crea un DataLoader con el Dataset
        streamlit_loader = DataLoader(streamlit_data, batch_size=1, shuffle=False)
        
        # Realizar la predicción
        predicted_labels = model.predict(streamlit_loader)

        predicted_label = predicted_labels[0]

        class_name = classnames[predicted_label]

        # Mostrar la predicción y la imagen
        st.write(f'### Clase predicha: {class_name}')
        st.image(image_file, caption='Imagen cargada', use_column_width=False)
        

if __name__ == "__main__":
    main()