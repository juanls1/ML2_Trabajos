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
from config.variables import Images_size, Images_types, Disp_Models, Models_paths


def main():
    # Configuración de la página
    st.set_page_config(page_title="ML2 - CNN", layout="centered")
    st.title("Clasificación de Imágenes con CNNs")
    
    # Mensaje de bienvenida
    with st.container():
        st.markdown("""
            ¡Bienvenidos a la aplicación web Canonist.ia de clasificación de imágenes del grupo compuesto por Alberto, Jorge, Nacho y Juan!
            Esta aplicación utiliza redes neuronales convolucionales (CNNs) para clasificar imágenes. Por favor, selecciona el modo de clasificación en la barra lateral y carga una imagen.
        """)

    # Configuraciones de la barra lateral
    with st.sidebar:
        st.header("Configuraciones")
        # Selector de modo de clasificación
        classification_mode = st.radio(
            "Modo de Clasificación:",
            ("Single-class", "Multi-class"),
            help="Selecciona 'Single-class' si deseas que la imagen se clasifique en una sola categoría. Elige 'Multi-class' para obtener múltiples posibles categorías."
        )
        
        # Selector del modelo
        model_option = st.selectbox(
            "Modelo a Utilizar:",
            Disp_Models,
            help="Selecciona el modelo de CNN que deseas usar para clasificar tu imagen."
        )
    
    # Carga de imagen y selección de modelo
    with st.container():
        image_file = st.file_uploader("Cargar Imagen", type=Images_types)
    
    if image_file is not None:
        with st.spinner('Procesando imagen...'):
            # Aquí iría todo el proceso de predicción basado en la elección de "Single-class" o "Multi-class"
            # Imaginemos que obtenemos el nombre de la clase predicha
            class_name = "Ejemplo de Clase"
            st.success(f'Clasificación completada: {class_name}')
            st.image(image_file, caption='Imagen Cargada', use_column_width=True)

if __name__ == "__main__":
    main()