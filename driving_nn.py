import numpy as np
import csv
import cv2
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from  keras.regularizers import l1
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
'''
Notes from walkthrough:
Goal is to drive a car around a track to generate training data 
from a simulated camera and use them to decide how to drive the car, 

Looking at the simulator. First, jump into training mode, record. 
You will probably need a GPU to train it. See 3:00 for how to bring up an
EC2 instance. Look for udacity-carnd ami in Northern California. Then select g2.2xlarge instance. 
For this project default security is sufficient.

the generated csv file does not have headings, the sample one does:
center image, left image, right image, steering, throttle, brake, speed

images: images are 320x160. The bottom 25 pixels are the hood of the car 
and the top 70 pixels are the sky and hills, so we may want to remove it later.

We can use all three, for now just use the center image

zip the data and scp it to the AWS instance

open the csv file using csv package
source of image file will be whole path. Will need to split it into root 
(which varies with location) and path (which does not) 

instead of using LeNet we could use nvidia's end-to-end deep learning for self-driving cars

for generator use fit_generator instead of fit

Instance ID is 065a62438f56adc9c

public IP is 54.219.185.59
'''
def process_image(img):
    img = cv2.GaussianBlur(img,(5, 5),0)
    return img


def read_data(root_dir):
    car_images = []
    steering_angles = []
    csv_file = "./" + root_dir + "/driving_log.csv"
    # create adjusted steering measurements for the side camera images
    correction = 0.19 # this is a parameter to tune, increased from initial 0.2
    threshold = 0.985 # only keep 10% of the steering values under +/- 0.15
    filter = 0.15 # filter value to remove middle values
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[3] == "steering":
                continue
            steering_center = float(row[3])
            # read in images from center, left and right cameras
            path=  "./" + root_dir + "/IMG/"
            # fill in the path to your training IMG directory
            token = "\\" # sample data uses forward slash, generated data uses backward slash
            if row[0].find("\\") == -1:
                token = "/"
            img_center = process_image(np.asarray(cv2.imread(path + row[0].strip().split(token)[-1])))
            img_left = process_image(np.asarray(cv2.imread(path + row[1].strip().split(token)[-1])))
            img_right = process_image(np.asarray(cv2.imread(path + row[2].strip().split(token)[-1])))

            # add images and angles to data set
            keep = True
            if abs(steering_center) < filter:
                pr_val = np.random.uniform()
                if pr_val < threshold:
                    keep = False
            if keep:
                steering_angles.append(steering_center)
                car_images.append(img_center)            
                steering_angles.append(steering_center +  correction)
                car_images.append(img_left)
                steering_angles.append(steering_center - correction)
                car_images.append(img_right)
    # augment the data so that we get left turn and right turn data equally
    # as car is moving clockwise, so we need counter clockwise data
    augmented_images = []
    augmented_steering = []
    for car_image, steering_angle in zip(car_images, steering_angles):
        augmented_images.append(car_image)
        augmented_steering.append(steering_angle)
        flipped_image = cv2.flip(car_image, 1)
        flipped_steering = steering_angle * -1.0
        augmented_images.append(flipped_image)
        augmented_steering.append(flipped_steering)
    return np.array(augmented_images), np.array(augmented_steering)


def simple_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

def simple_model_with_normalization_and_cropping():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    #crop the top 70 px and bottom 25px to eliminate parts we don't care about
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def simple_convolutional_model(p_dropout):
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    #crop the top 70 px and bottom 25px to eliminate parts we don't care about
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(3,1,1, activation='relu')) # color filter
    model.add(MaxPooling2D(pool_size=(2,2))) # 2,2 causes the model to fail on negative dimension value
    model.add(Dropout(p_dropout))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def simple_mixed_model(p_dropout):
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    #crop the top 70 px and bottom 25px to eliminate parts we don't care about
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(3,1,1, activation='relu')) # color filter
    model.add(MaxPooling2D(pool_size=(2,2))) # 2,2 causes the model to fail on negative dimension value
    model.add(Dropout(p_dropout))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(1))
    return model

def simple_mixed_model2(p_dropout):
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    #crop the top 70 px and bottom 25px to eliminate parts we don't care about
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(3,1,1, activation='relu')) # color filter
    model.add(MaxPooling2D(pool_size=(2,2))) # 2,2 causes the model to fail on negative dimension value
    model.add(Dropout(p_dropout))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(p_dropout))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(1))
    return model

def leNet(p_dropout):
    model = Sequential()
    #crop the top 70 px and bottom 25px to eliminate parts we don't care about
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(p_dropout))
    model.add(Convolution2D(16,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(p_dropout))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nvidia_model(p_dropout, p_regularize):
    # devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    #crop the top 70 px and bottom 25px to eliminate parts we don't care about
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(3,1,1, activation='relu', W_regularizer=l1(p_regularize))) # color filter
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu', W_regularizer=l1(p_regularize)))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu', W_regularizer=l1(p_regularize)))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu', W_regularizer=l1(p_regularize)))
    model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l1(p_regularize)))
    model.add(MaxPooling2D(pool_size=(1,2))) # 2,2 causes the model to fail on negative dimension value
    model.add(Dropout(p_dropout))
    model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l1(p_regularize)))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Dropout(p_dropout))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l1(p_regularize)))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def generator(x, y, batch_size=32):
    num_samples = len(x)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_x = x[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]
            X_train = np.array(batch_x)
            y_train = np.array(batch_y)
            yield (X_train, y_train)


import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
def main():
    p_dropout = 0.5
    p_regularize = 0.001
    X_train, y_train = shuffle(read_data("IMG_TRAIN4"))
    shape_x = X_train.shape
    shape_y = y_train.shape
    n, bins, patches = plt.hist(np.asarray(y_train*100), 50, normed=True)
    
    plt.xlabel('steering')
    plt.ylabel('magniture')
    plt.title(r'Adjusted steering values')
    plt.grid(True)
    
    plt.show()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    train_generator = generator(X_train, y_train, batch_size=32)
    validation_generator = generator(X_val, y_val, batch_size=32)
    # use letNet model
    model = nvidia_model(p_dropout, p_regularize)
    # use of optimizers, see https://keras.io/optimizers/
    # according to one student, a learning rate of 0.001 yields
    # smoother turns, where 0.0001 yields sharper turns if your
    # car isn't turning far enough
    model.compile(optimizer='adam', loss='mse', lr=0.001)
    model.fit_generator(train_generator, samples_per_epoch=len(X_train), 
            validation_data=validation_generator, 
            nb_val_samples=len(X_val), nb_epoch=4)
    model.save('model.h5')
if __name__ == "__main__":
    main()