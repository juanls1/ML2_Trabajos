# Constants necessary for the project

# Constants used for displaying a random sample of images in the notebook for training
Number_random_images = 64
Number_rows = 8
Number_cols = 8
Figsize = (15, 15)
Fontsize = 8

# Constants used for displaying a random sample of images in the notebook for validation
Number_random_images_val = 64
Number_rows_val = 8
Number_cols_val = 8
Figsize = (15, 15)
Fontsize = 8


# Constants used for loading the data (they affect the performance and functioning of the model)
Batch_size = 32 
Images_size = 224   # ResNet50 requires 224x224 images

# Constants used for the model
Model_used = 'resnet50'
Learning_rate = 1e-4
Number_epochs = 1
Criterion = 'cross_entropy'
Optimizer = 'Adam'
Model_name = 'resnet50-1epoch' # Name of the model to save

# Constants used for the prediction
Model_loaded = 'resnet50-1epoch' # Name of the model to load
Model_used_load = 'resnet50' # Model used for the loaded model

# Constants used for the streamlit app
Images_types = ['png', 'jpg', 'jpeg']
Disp_Models = ["Modelo 1 (1 epoch)", "Modelo 2 (4 epochs)"]
Models_paths = ['C:/Users/juanl/Documents/Master/2o cuatri/ML2/Codes/Entregas/ML2_Trabajos/2nd project/models/resnet50-1epoch', 
                'C:/Users/juanl/Documents/Master/2o cuatri/ML2/Codes/Entregas/ML2_Trabajos/2nd project/models/resnet50-1epoch']
