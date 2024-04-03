import streamlit as st
from PIL import Image
import torch
from data_loader import num_classes, classnames
from torch.utils.data import DataLoader
from local_functs import CustomImageDataset, load_model
from torchvision import transforms



def main():
    st.title("Clasificación de imágenes con CNNs")
    st.markdown("¡Bienvenidos a la aplicación web de clasificación de imágenes del grupo compuesto por Alberto, Jorge, Nacho y Juan!")
    
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

        img_size = 224

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
        st.write(f'Clase predicha: {class_name}')
        st.image(image_file, caption='Imagen cargada', use_column_width=True)
        

if __name__ == "__main__":
    main()