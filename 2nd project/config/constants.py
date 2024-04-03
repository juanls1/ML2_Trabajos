# Constants necessary for the project

# Constants used for displaying a random sample of images in the notebook
Number_random_images = 64
Number_rows = 8
Number_cols = 8
Figsize = (15, 15)
Fontsize = 8

# Constants used for loading the data (they affect the performance and functioning of the model)
Batch_size = 32 
Images_size = 224   # ResNet50 requires 224x224 images

# Constants used for the streamlit app
Images_types = ['png', 'jpg', 'jpeg']
Disp_Models = ["Modelo 1 (1 epoch)", "Modelo 2 (4 epochs)"]
Models_paths = ['C:/Users/juanl/Documents/Master/2o cuatri/ML2/Codes/Entregas/ML2_Trabajos/2nd project/models/resnet50-1epoch', 
                'C:/Users/juanl/Documents/Master/2o cuatri/ML2/Codes/Entregas/ML2_Trabajos/2nd project/models/resnet50-1epoch']
