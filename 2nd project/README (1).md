# Classiffying AI for environment recognition

## Creation of an AI that is capable to receive an input picture and classify the picture according to its environment(living room, forest, coast,...)

The project mainly consists on the training of different CNNs through the use of pretrained models, in order to get a model that can efficiently classify the picture environment. 
The training of models is connected with Weights and Biases, so all the metrics of every model trained will be reported in W&B.

**The different classes of environments that IA has been trained to properly predict can be seen in the data directory** 

Along with the AI model, the project also includes the functionality of launching an streamlit app for the use of the AI. It enables the user to upload a picture and classify it using some of the available models in a way more user-friendly manner.


## Previous requirements 📋

 1. Clone the repository

```
git clone https://github.com/juanls1/ML2_Trabajos/
```

(This whole project corresponds with the 2nd project directory of the repository )

 2. Create a venv & install the necessary libraries 

```
conda create --name CNN

pip install -r requirements.txt

```

3. If necessary, update ```.gitignore``` with your own sensitive files


## Folder Explanation :file_folder: 

 + **config:** Folder containing the .py file with the parameters of the CNN that will be trained. It includes, pretrained model name, number of epochs, learning rate,... 

 + **data:** Folder containing all the inputs used for the model. It is split in 3 sets, one for training of the model, other for its validation(checking the overfitting), and a smaller one for testing the model in the end.

 + **models** Folder containing the weights, architectures and all the information about the already trained CNNs.

 + **src:** Folder containing the core of the model. It includes a streamlit directory with the needed .py to launch the streamlit app. It also inncludes a utils directory including some of the fucntions and the CNN class required for the models training.
 And mainly, it includes the model_training.py script whose execution performs the training of the AI model.

 + **wandb:** As all the info of the models is reported to W&B, W&B sends some logs about the information from each model saved in the page. That info is contained in this repository.


## What can be done with the repository  

 ### Training a model

1. Tune the parameters of the model you are willing to train in the variables.py script

2. Execute the model
```
 python model_training.py
```
3. Review the training of the model through W&B

After executing the model_training cript, an URL that guides the user to the W&B report about the model report will be shown.



### Running the streamlit app to classify new images
```
python src/streamlit/app.py

```



 
## Current Improvement Work 🔧

During the development of the project some ideas to try and achieve more accurate models came up, however they have not been implemented so far due to lack of time.

It must be pointed out that the best model trained so far achieved about a 95% accuracy in both the training dataset and the validation dataset. 

### 1.Multi-Class Output Models:
 When the 5 % of pictures in which the models were misclassifying was analysed, a pattern was detected. In some pictures, two of the possible environments the model can classify coul be appreciated, such as pictures of highway near the sea, in which the model struggled to decide between the coast label or the highway label. In the future, we will try to tune the model, so in this cases it can return the two most likely classes.

### 2.Data Augmentation:
 A work of rotating, scaling, and applying these type of transformations to the pictures in the dataset could be applied in order to increase the size of the dataset

### 3.Second Level Models
 An idea is to introduce the pictures for which the AI model returns lower levels of confidence in 'submodels'. Those models would have been trained exclusively to determine if a picture belongs to a given class or not. The final prediction would be decided according to the submodel that returns a higher level of confidence for predicting positively. This technique may increase the accuracy of our models.

### 4.HugginFace models
 In the future it will be tried to train models using HugginFace, what should improve the models accuracy, as those are more modern and novel models than the ones used so far.
