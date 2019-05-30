# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 

python drive.py model.h5


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
I started from a LeNet Architecture and then used NVIDIA's Deep Learning Neural Network for Autonomous Car with a few Modifications.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 64-67) 

The model includes RELU layers to introduce nonlinearity (code lines 64-67), and the data is normalized in the model using a Keras lambda layer (code line 61). 
It Also contains a Image Cropping Layer to decrease the time of training(line 63)
It also contains Dropout layers to decrease the overfitting of the Model(lines 69-73)
3 Fully Connected Layers (lines 69-73)

#### 2. Attempts to reduce overfitting in the model

The model contains 3 dropout layers in order to reduce overfitting (model.py lines69-73). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 77-80). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 77).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also used a training data at few particular sections of the track where there are no lane lines in Track one. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to give enough information to the model to train it to drive at the centre of th road.Collect enough data and supply it the Neural Network.

My first step was to use a neural network model which was just a random layers layed out to test that code is working and then once i got that
MY Second step was to build a CNN similar to LeNet,I thought this model might be appropriate because it was powerful and popular for CNN's 
And once i got the loss from it and tested it out on the track i wanted a more powerful Architecture.
For which i used NVIDIA's Model with a few Modifications which gave the best results overall

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my NVIDIA's Modified model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that by adding the Dropout layers the loss of both the Training and Validation Set is less

Then I tested it out 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track during intial testing of the model to improve the driving behavior in these cases, I added more data that is recovery laps and added few dropout layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 59-74) consisted of a convolution neural network with the following layers and layer sizes ...
First layer was normalizing layer of the input images
Second layer was cropping the Normalized images and then the Subsequent Layers were of 4 Convolutional Layers of 24,36,48,64 with size of 5 by 5 except the last one with 2 by 2. With each layer having an activation of RELU for non-linearity.
Then the last few layers were Fullyconnected layers with in between Dropout layers included in it.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the centre.
I then recorded the vehicle recovering from the side where there is no lane line present. 
YOu could see the images of the Recovery laps at home/backups/img

To augment the data sat, I also flipped images and angles thinking that this would give the model a more generic view of driving and essentially more data to train.

After the collection process, I had 2 sets of Data Which one was normal driving and the other was recover laps at the sections where there were no Lane lines. I then preprocessed this data by using the Lambda layer.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by running at higher epochs and the loss was increasing which was a result of overfitting so i finalised it at 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.

At the the model was able to make the car Drive Autonomously. 
I recorded the Video of it you can see it at Video.mp4
