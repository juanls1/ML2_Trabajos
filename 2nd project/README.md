# CANONIST.IA: Classiffying AI for environment recognition

### Creation of an AI that is capable of receiving an input picture and classify the picture according to its environment 

The project mainly consists on the training of different CNNs through the use of pretrained models, in order to get a model that can efficiently classify the picture environment. 
The training of models is connected with Weights and Biases, so all the metrics of every model trained will be reported in W&B.

**The different classes of environments that IA has been trained to properly predict can be seen in the data directory** 

Along with the AI model, the project also includes the functionality of launching an streamlit app for the use of the AI. It enables the user to upload a picture and classify it using some of the available models in a way more user-friendly manner.


## Previous requirements 📋

 1. Clone the repository

```
git clone https://github.com/juanls1/ML2_Trabajos.git
```

(This whole project corresponds with the 2nd project directory of the repository )

 2. Create a venv & install the necessary libraries 

```
cd "ML2_Trabajos/2nd Project"
python -m venv ML_deep_learning_Entrega
ML_deep_learning_Entrega\Scripts\activate
pip install -r requirements.txt

```

3. If necessary, update ```.gitignore``` with your own sensitive files



> [!IMPORTANT]
> If you want to train the models with the **GPU**, you'll have to install [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and the respectiove versions of [Torch and torchivision](https://pytorch.org/) manually, as the versions from the requirements.txt are for **CPU** training


## Folder Explanation :file_folder: 

 + **config:** Folder containing the .py file with the parameters of the CNN that will be trained. It includes, pretrained model name, number of epochs, learning rate,... 

 + **data:** Folder containing all the inputs used for the model. It is split in 3 sets, one for training of the model, other for its validation(checking the overfitting), and a smaller one for testing the model in the end.

 + **models:** Folder containing the weights, architectures and all the information about the already trained CNNs.

 + **src:** Folder containing the core of the model. It includes a streamlit directory with the needed .py to launch the streamlit app. It also inncludes a utils directory including some of the fucntions and the CNN class required for the models training.
 And mainly, it includes the model_training.py script whose execution performs the training of the AI model.

 + **wandb:** As all the info of the models is reported to W&B, W&B sends some logs about the information from each model saved in the page. That info is contained in this repository.

> [!IMPORTANT]
> The models have not been uploaded to the models folder because they are heavier than 100MB, so GitHub does not allow it. In order to run the streamlit app with the models we have trained, they will need to be downloaded from the following Google Drive folder, and upload them into the models folder.
Google Drive Folder: https://drive.google.com/drive/folders/1FIgMu_dcUOpQlBKDgDgYiMXHL3J-99we?usp=sharing

## What can be done with the repository  

 ### Training a model

1. Tune the parameters of the model you are willing to train in the variables.py script

2. Execute the model
```
 python model_training.py
```
3. Review the training of the model through W&B
```
 https://wandb.ai/mbd2023/ML2-CNN-PROJECT/table?nw=nwuser<your_user>
```

After executing the model_training cript, an URL that guides the user to the W&B report about the model report will be shown.



### Running the streamlit app to classify new images
```
streamlit run src/streamlit/app.py

```


## Best Models
Below, we describe briefly, the models that can be found in the Google Drive folder and that have been considered to be the best models found so far

**1.Resnet50 trained with 50 epochs:** The first model availabe is a simple resnet model that was the one we used in the first class presentation.
 *Train Accuracy:90.55%
 *Validation Accuracy:94.27%

 **2.Resnext101 trained with Cross-Entropy Loss:** This is considered to be the best model trained. It is trained with CPU and with the default Cross-Entropy Loss function. As this one is considered the best model, it is evaluated also in the test set.
 *Train Accuracy:95.24%
 *Validation Accuracy:94.33%
 *Test Accuracy:92.89%

**3.Resnext101 trained with Custom Loss:** As it will be explained below in more detail, it was tried to create a new loss function and train models with it instead of Cross-Entropy. This one is the best among all the models trained with this customized metric:
 *Train Accuracy:92.36%
 *Validation Accuracy:95.92%

 **4.Convnext model trained with GPU(NOT AVAILABLE):** Another phase of the project was training models with the GPU, so, as the training times would be shorter, more complex models could be trained. This one was the best model obtained with GPU, however due to some error, the last two layers of its classificator have not been loaded, so it does not work.
 *Train Accuracy:94.24%
 *Validation Accuracy:96.37%
 ****
 
## Current Improvement Work 🔧

During the development of the project some ideas to try and achieve more accurate models came up, however they have not been implemented so far due to lack of time.

It must be pointed out that the best model trained so far achieved about a 95% accuracy in both the training dataset and the validation dataset. 

### 1.Multi-Class Output Models
 When the 5 % of pictures in which the models were misclassifying was analysed, a pattern was detected. In some pictures, two of the possible environments the model can classify could be appreciated, such as pictures of highway near the sea, in which the model struggled to decide between the coast label or the highway label. In the future, we will try to tune the model, so in this cases it can return the two most likely classes.

### 2.Data Augmentation
 A work of rotating, scaling, and applying these type of transformations to the pictures in the dataset could be applied in order to increase the size of the dataset

### 3.Second Level Models
 An idea is to introduce the pictures for which the AI model returns lower levels of confidence in 'submodels'. Those models would have been trained exclusively to determine if a picture belongs to a given class or not. The final prediction would be decided according to the submodel that returns a higher level of confidence for predicting positively. This technique may increase the accuracy of our models.

### 4.HugginFace models
 In the future it will be tried to train models using HugginFace, what should improve the models accuracy, as those are more modern and novel models than the ones used so far.

### 5.Training with new Loss Function
It was attempted to train models using a loss function specifically designed for this problem. As it was mentioned earlier, in some of the pictures our models misclassified, two potential classes could be appreciated. 
Due to that, we tried to create a new loss function that did not penalize so much a missclasification in the cases that the true class was the second option for the model. 
The results obtained were quite similar to the ones obtained with the Cross-Entropy Loss Function, however, in the future we will follow this line of thought in order to get more accurate models.

Nonetheless, with a threshold of 0.95, better results were acheived. Moreover, it is important to highlight that the custom loss model has space for improvement (only 10 epochs, with CPU, only 8 unfreezed layers and custom loss threshold of 0.5 that could be changed in order to train better). (This last paragraph and the change of threshold 
to 0.95 was done on tuesday, in order to perform the last tests)

