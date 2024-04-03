from cnn import load_data

# Load data 
train_dir = 'C:/Users/juanl/Documents/Master/2o cuatri/ML2/Codes/Entregas/ML2_Trabajos/2nd project/data/training'
valid_dir = 'C:/Users/juanl/Documents/Master/2o cuatri/ML2/Codes/Entregas/ML2_Trabajos/2nd project/data/validation'

train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                    valid_dir, 
                                                    batch_size=32, 
                                                    img_size=224) # ResNet50 requires 224x224 images

classnames = train_loader.dataset.classes