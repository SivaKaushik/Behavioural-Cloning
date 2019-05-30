import csv
import cv2
import numpy as np
from scipy import ndimage

lines = []
lines_1 = []
#Reading the Sample Data Given
with open("driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#Reading the Data Collected by myself in Training Mode which contains Recovery Laps
with open("driving_log_1.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines_1.append(line)
images = []
measurements = []
#Adding images and corresponding Steering angle measurements from the Sample Data
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/'+filename
    image =ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
#Adding images and Corresponding steering angle measurements from the data i collected
for line in lines_1:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '/home/backups/IMG/'+filename
    image =ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
 
#Creating more data by augmenting the images and their respective measurements
augmented_images = []
augmented_measurements = []
for image in images:
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image,1))
    
for measurement in measurements:
    augmented_measurements.append(measurement)
    augmented_measurements.append(measurement*-1.0)

# Changing the Lists into Numpy Arrays for training the neural network   
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
#importing the required functions
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
# The Neural Network based on NVIDIA's Autonomous Testing given in the Course with few dropout layers added to decrease overfitting
model = Sequential()
#Normalizing the Input Images
model.add(Lambda(lambda x: (x/255.0)-0.5,input_shape = (160,320,3)))
#Cropping the input images to decrease the Training time and taking out unnecessary parts of the image
model.add(Cropping2D(cropping=((75,25),(0,0))))
model.add(Conv2D(24,(5,5),strides=(2,2),activation ='relu'))
model.add(Conv2D(36,(5,5),strides=(2,2),activation ='relu'))
model.add(Conv2D(48,(5,5),strides=(2,2),activation ='relu'))
model.add(Conv2D(64,(2,2),activation ='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(1))

#Using Mean Square Error to Calculate the Loss and using Adam's Optimizer
model.compile(loss='mse',optimizer='adam')

#Training the Neural Network and creating a Validation Set from the Training set which is 20% of the Training Set
model.fit(X_train,y_train,validation_split = 0.2,shuffle = True,epochs=4)

#Saving the Neural Network            
model.save('model.h5')
    