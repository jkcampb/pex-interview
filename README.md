Pex Machine Learning Interview
==============================

IMPORTANT NOTE
-----------
The trained model is a 300MB file, so git-lfs is needed in order to download it along with the rest of the repository. Info on getting git-lfs can be found here: https://git-lfs.github.com/

What is it?
-----------
A python script that will take an image path as an argument and return whether or not that image is of the indoors or outdoors based on a pre-trained convolutional neural network model. 
This repo also contains Jupyter notebooks that show how the train and test images were acquired and how the CNN was trained.

How to use
-----------
1. Run `pip install -r requirement.txt` to install dependencies in a python 3 environment.
2. Run `python3 indoor_outdoor.py {path-to-image}`

Notebooks
-------------
#### 0_TFRecords_Get_Target_YoutubeId.ipynb
Shows how data was parsed from http://research.google.com/youtube8m/explore.html to acquire relevant labels and real Youtube IDs.
#### 1_Get_Youtube_Frames.ipynb 
Uses data output from previous notebook to pull the requested number of frames from the Youtube ID and saves them to the relevant folder in the **data** directory.
#### 2_Tensorflow_Training.ipynb
Uses the frames saved from the previous notebook to train a CNN model using Tensorflow. That model is then saved and used as the model for **indoor_outdoor.py**