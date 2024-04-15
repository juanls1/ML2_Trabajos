# Constants necessary for the project
import sys
from pathlib import Path
import os

# Obtener la ruta absoluta de la carpeta que contiene el módulo
root_dir = Path(__file__).resolve().parent.parent

# Agregar la ruta de la carpeta al sys.path
sys.path.append(str(root_dir))


# Constants used for displaying images (visualization.ipynb, training set sample)

Number_random_images = 64
Number_rows = 8
Number_cols = 8
Figsize = (15, 15)
Fontsize = 8



# Constants used for displaying images (visualization.ipynb, validation set sample)  

Number_random_images_val = 64
Number_rows_val = 8
Number_cols_val = 8
Figsize = (15, 15)
Fontsize = 8



# Constants for loading data (they affect the performance and functioning of the model)

Batch_size = 128  # 2985 images in the training set, 1500 in the validation set
Images_size = 224  

# ResNet50: 224x224 images
# ResNet18: ??x?? images



# Constants used for the model     (model_training.py)

Saved_epochs = 5
extra_models = ['personal_cnn'] # It has to be done. A cnn from scratch. If we have more models, they will be added here
classification_models = ['alexnet','convnext_base','convnext_large','convnext_small','convnext_tiny','densenet121','densenet161','densenet169','densenet201',
                         'efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7',
                         'efficientnet_v2_l','efficientnet_v2_m','efficientnet_v2_s','googlenet','inception_v3','maxvit_t','mnasnet0_5','mnasnet0_75','mnasnet1_0',
                         'mnasnet1_3','mobilenet_v2','mobilenet_v3_large','mobilenet_v3_small','regnet_x_16gf','regnet_x_1_6gf','regnet_x_32gf','regnet_x_3_2gf',
                         'regnet_x_400mf','regnet_x_800mf','regnet_x_8gf','regnet_y_128gf','regnet_y_16gf','regnet_y_1_6gf','regnet_y_32gf','regnet_y_3_2gf',
                         'regnet_y_400mf','regnet_y_800mf','regnet_y_8gf','resnet101','resnet152','resnet18','resnet34','resnet50','resnext101_32x8d','resnext101_64x4d',
                         'resnext50_32x4d','shufflenet_v2_x0_5','shufflenet_v2_x1_0','shufflenet_v2_x1_5','shufflenet_v2_x2_0','squeezenet1_0','squeezenet1_1','swin_b',
                         'swin_s','swin_t','swin_v2_b','swin_v2_s','swin_v2_t','vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn','vit_b_16',
                         'vit_b_32','vit_h_14','vit_l_16','vit_l_32','wide_resnet101_2','wide_resnet50_2'] # Obviously we do not have to run all. Do a research, test the bests
Model_used = 'resnet18'
Learning_rate = 1e-4
Unfreezed_layers = 0
Number_epochs = 3
Criterion = 'CrossEntropyLoss'   # 'CustomLoss' or 'CrossEntropyLoss'
Optimizer = 'Adam'
threshold = 0.2
# Otro optimizer : torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # SGD with momentum
Model_name = f"{Model_used}-LR_{Learning_rate}-NE_{Number_epochs}-UL_{Unfreezed_layers}-C_{Criterion}-O_{Optimizer}" # Name of the model to save



# Constants used for the prediction  (visualization.ipynb)

Model_loaded = Model_name # Name of the model to load
Model_used_load = Model_used # Model used for the loaded model



# Constants used for the streamlit app

Images_types = ['png', 'jpg', 'jpeg']
Disp_Models = ["Simpler Model(Resnet50)", 
               "CPU Best Model(Resnext101)",
               "Best model with CustomLoss(Resnext101)",
		"GPU Best Model"]
Models_dir = os.path.join(root_dir, 'models')
Models_paths = [os.path.join(Models_dir, 'resnet50-LR_0.0001-NE_50-UL_3-C_CrossEntropyLoss-O_Adam'), 
                os.path.join(Models_dir, 'resnext101_32x8d-LR_0.0001-NE_30-UL_13-C_CrossEntropyLoss-O_Adam'),
                os.path.join(Models_dir, 'resnext101_32x8d-LR_0.0001-NE_10-UL_8-C_CustomLoss-O_Adam'),
		os.path.join(Models_dir, 'convnext_small-LR_1e-05-NE_50-UL_7-C_CrossEntropyLoss-O_Adam')]
