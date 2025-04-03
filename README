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
        |---- src_train_test  => sourcecode for training and testing the NN model
        |---- src_runtime  => sourcecode for running the complete program
        |---- src_runtime_async  => sourcecode for running the complete program in asynchronous mode

Programs scr_train_test
requires installations:
sudo apt install python3-scikit-learn
sudo apt install python3-scikit-image
sudo apt install python3-tqdm
sudo apt install python3-torch
sudo apt install python3-torchvision
sudo apt install python3-seaborn
sudo apt install python3-opencv-python
sudo apt install python3-numpy

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

4_Demo.py
Standalone desktop demo to assess performance on real images
- loads the NN model from 2_TrainResnet.py
- takes images from the camera
- extracts the features from the frames, detects if object is present and has the NN model classify the frame


Programs scr_runtime
Contains the original code used for demonstration
To run without internet, you must run with an ethernet connection first because the program needs the default resnet model
it will download the model on "https://download.pytorch.org/models/resnet34-b627a593.pth"
to "/home/xxxx/.cache/torch/hub/checkpoints/resnet34-b627a593.pth"

Programs scr_runtime_async
WIP: code to run in asynchronous mode
sudo apt install python3-aiomqtt
sudo apt install python3-opencv-python
sudo apt install python3-numpy
sudo apt install python3-pigpio

Framework:
Motion Detection (detect_motion)
__Monitors the video feed for movement and triggers image capture.

Image Capture (capture_image)
__Increases illumination, captures an image, and triggers classification.

Image Classification (evaluate_image)
__Uses multiprocessing to classify the image in a separate process.

Sorting & LED (sort_object)
__Moves the servo and turns on the correct LED.

MQTT Publishing (mqtt_publish)
__Sends classification results over MQTT.

asyncio.gather()
__Runs all tasks concurrently.
