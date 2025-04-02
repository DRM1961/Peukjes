# Peukjes

Folders and content
Peukjes |---- Data => contains a scaler and the NN model for runtime testing
        |---- Debug => contains csv file with performance data on datasets
        |---- Images 
        |     |---- DataSet => generated augmented images for training
        |     |---- Extract => generated images for augmentation 
        |     |---- Originals
        |           |---- GeenPeukjes => original training images of some garbage in a cup
        |           |---- Peukjes     => original training images of peukjes in a cup 
        |---- src  => sourcecode
              
Programs
1_CreateDataset.py 
Performs basic augmentation on a series of original GOOD/BAD images.
Using an "empty" scene as reference:
- it aligns each image with the reference
- it extracts the bounding box around the largest difference with the reference
- it rotates the box under multiple angles and merges it with the empty scene 
- and saves all the images in a folder that can be used to train an AI classification model

2_TrainResnet.py
Trains a neural network on the difference between a peukje and not a peukje
- takes the images from 1_CreateDataset
- divides them up in a training and a testing set
- uses a ResNet premodeled feature extractor reducing the images to an array if "n" feature-values
- uses a simple classifier network with two categories (peukje or not)
- feeds the features from all the training images to the network a number of times for model optimization
- calculates some statistics and saves the NN model

3_CheckAllImages.py
Checks the classification performance on all the images in the dataset and saves the results for review
- loads the NN model from 2_TrainResnet.py
- takes the images from 1_CreateDataset
- extracts the features from each image and has the NN model classify the image
- the raw data of classification and probabilities for GOOD/BAD are saved in a csv file for review


