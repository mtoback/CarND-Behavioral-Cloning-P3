import argparse
import csv
import cv2
from datetime import datetime
import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, BatchNormalization
from keras.layers.core import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.misc import imresize
import sys
import time
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

def read_data(root_dir, p_correction, p_threshold, p_filter):
    car_images = []
    steering_angles = []
    csv_file = "./" + root_dir + "/driving_log.csv"
    # create adjusted steering measurements for the side camera images
    correction = p_correction # amount to add to/subtract from right/left steer
    threshold = p_threshold # only keep higher % of the steering values under +/- some threshold (so 99% means keep 1%)
    filter = p_filter # filter value to remove middle values
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
    model.add(BatchNormalization(input_shape=(160,320,3), axis=1)) # lambda has issues with unicode, use BatchNormalization instead
    model.add(Flatten())
    model.add(Dense(1))
    return model

def simple_model_with_normalization_and_cropping():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(160,320,3), axis=1)) # lambda has issues with unicode, use BatchNormalization instead
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
    model.add(Convolution2D(24,5,5, activation='relu')) # color filter
    model.add(MaxPooling2D(pool_size=(2,2))) # 2,2 causes the model to fail on negative dimension value
    model.add(Dropout(p_dropout))
    model.add(Convolution2D(36,5,5, activation='relu')) # color filter
    model.add(MaxPooling2D(pool_size=(2,2))) # 2,2 causes the model to fail on negative dimension value
    model.add(Dropout(p_dropout))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def simple_convolutional_model2(p_dropout):
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    #crop the top 70 px and bottom 25px to eliminate parts we don't care about
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5, activation='relu')) # color filter
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
    model.add(Dense(100))
    model.add(Dense(1))
    return model

def simple_mixed_model2(p_dropout, p_regularize):
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    #crop the top 70 px and bottom 25px to eliminate parts we don't care about
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(3,1,1, activation='relu')) # color filter
    model.add(MaxPooling2D(pool_size=(2,2))) # 2,2 causes the model to fail on negative dimension value
    model.add(Dropout(p_dropout))
    model.add(Flatten())
    model.add(Dense(120, W_regularizer=l2(p_regularize)))
    model.add(Dense(1))
    return model

def simple_mixed_model3(p_dropout):
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
    model.add(Dense(100))
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
    model.add(BatchNormalization(input_shape=(160,320,3), axis=1)) # lambda has issues with unicode, use BatchNormalization instead
    #model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    #crop the top 70 px and bottom 25px to eliminate parts we don't care about
    model.add(Cropping2D(cropping=((35,12),(0,0))))
    #model.add(Convolution2D(3,1,1, activation='relu')) # color filter
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
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

model_list = ['simple_model','simple_model_with_normalization_and_cropping',
                              'simple_convolutional_model',
                              'simple_convolutional_model2',
                              'simple_mixed_model',
                              'simple_mixed_model2',
                              'simple_mixed_model3',
                              'leNet','nvidia_model']
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
def show_dist(directory, y_train, p_correction, p_threshold, p_filter):
    n, bins, patches = plt.hist(np.asarray(y_train*100), 50, normed=True)
    
    plt.xlabel('steering')
    plt.ylabel('magniture')
    plt.title('Adjusted steering values for %s, corr=%03f,\nthresh=%03f, filt=%03f' % (directory, p_correction, p_threshold, p_filter))
    plt.grid(True)
    plt.show()
    
def parser_error(p, msg):
    print(msg)
    p.print_usage()
    sys.exit(-1)

def show_history(history_object):
    
    print(history_object.history.keys())    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('images=%s, model=%s, drop=%05f,\nreg=%05f,lr=%05f' % 
              (image_dir, model_name, p_dropout, p_regularize, learning_rate))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='Remote Driving Deep Learning')
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help='names of model to train with'
    )
    parser.add_argument(
        '--images',
        required=True,
        type=str,
        help='folder in this directory containing the training images'
    )
    parser.add_argument(
        '--show_dist',
        type=bool,
        default=False,
        help='show distribution of training data'
    )
    parser.add_argument(
        '--show_hist',
        type=bool,
        default=False,
        help='show training thread of training data'
    )
    parser.add_argument(
        '--dropout_rate',
        required=True,
        type=float,
        help='dropout rate to be applied to model. Should be between 0 and 1.0'
    )
    parser.add_argument(
        '--regularize',
        required=True,
        type=float,
        help='regularization constant. Should be a positive, probably small (< 0.01) value'
    )
    parser.add_argument(
        '--learning_rate',
        required=True,
        type=float,
        help='learning rate constant for model. Should be a positive, probably small (< 0.01) value'
    )
    parser.add_argument(
        '--n_epochs',
        default=5,
        type=int,
        help='number of epochs to train the model. Should be a positive value'
    )
    parser.add_argument(
        '--correction',
        required=True,
        type=float,
        help='plus/minus value be applied to right/left steering values. Should be between 0 and 1.0'
    )
    parser.add_argument(
        '--threshold',
        required=True,
        type=float,
        help='percent values of middle steering values to be dropped'
    )

    parser.add_argument(
        '--filter',
        required=True,
        type=float,
        help='band of values for decimation of data be applied to center values. Should be between 0 and 1.0'
    )

    args = parser.parse_args()
    if args.images is None or len(args.images)<1:
        parser_error(parser,"need to specify images directory")
    if not os.path.exists(args.images):
        parser_error(parser,"images directory does not exist: %s" % args.images)
    
    if args.model is None or len(args.model)<1:
        parser_error(parser, "need to specify model to use")

    if args.model not in model_list:
        parser_error(parser, "model %s needs to be one of %s" %(args.model, model_list))
        
    
    if args.dropout_rate >= 1.0 or args.dropout_rate < 0.0:
        parser_error(parser, "dropout rate needs to be value between 0.0 and 1.0, 1.0 meaning no dropout")
    
    if args.regularize >= 1.0 or args.regularize < 0.0:
        parser_error(parser, "regularization parameter needs to be value between 0.0 and 1.0, 0.0 meaning no regularization")

    if (args.learning_rate >= 1.0) or (args.learning_rate <= 0.0):
        parser_error(parser, "learning rate needs to be value between 0.0 and 1.0, 0.0 meaning no learning")

    if args.n_epochs >= 1000 or args.learning_rate <= 0:
        parser_error(parser, "number of epochs needs to be positive value between 1 and 1000")
    if args.correction >= 1.0 or args.correction < 0.0:
        parser_error(parser, "correction needs to be value between 0.0 and 1.0, 1.0 meaning no dropout")
    if args.threshold >= 1.0 or args.threshold < 0.0:
        parser_error(parser, "threshold rate needs to be value between 0.0 and 1.0, 1.0 meaning no dropout")
    if args.filter >= 1.0 or args.filter < 0.0:
        parser_error(parser, "filter rate needs to be value between 0.0 and 1.0, 1.0 meaning no dropout")
    
    return {"images": args.images,
            "dropout": args.dropout_rate,
            "regularize": args.regularize,
            "learning_rate": args.learning_rate,
            "n_epochs": args.n_epochs,
            "model": args.model,
            "show_dist": args.show_dist,
            "show_hist": args.show_hist,
            "correction": args.correction,
            "threshold": args.threshold,
            "filter": args.filter}
import pickle

def main():
    params = get_args()
    p_dropout = params["dropout"]
    p_regularize = params["regularize"]
    image_dir = params["images"]
    learning_rate = params["learning_rate"]
    n_epochs = params['n_epochs']
    model_name = params["model"]
    p_correction = params["correction"]
    p_threshold = params["threshold"]
    p_filter = params["filter"]
    X_train, y_train = shuffle(read_data(image_dir, p_correction, p_threshold, p_filter))
    if params["show_dist"]:
        show_dist(image_dir, y_train,p_correction, p_threshold, p_filter)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    train_generator = generator(X_train, y_train, batch_size=32)
    validation_generator = generator(X_val, y_val, batch_size=32)
    
    model = None
    if model_name == "simple_model":
        model = simple_model()
    elif model_name == "simple_model_with_normalization_and_cropping":
        model = simple_model_with_normalization_and_cropping()
    elif model_name == "simple_convolutional_model":
        model = simple_convolutional_model(p_dropout)
    elif model_name == "simple_convolutional_model2":
        model = simple_convolutional_model2(p_dropout)
    elif model_name == 'simple_mixed_model':
        model = simple_mixed_model(p_dropout)
    elif model_name == 'simple_mixed_model2':
        model = simple_mixed_model2(p_dropout, p_regularize)
    elif model_name == 'simple_mixed_model3':
        model = simple_mixed_model3(p_dropout)
    elif model_name == 'leNet':
        model = leNet(p_dropout)
    elif model_name == "nvidia_model":
        model = nvidia_model(p_dropout, p_regularize)
    else:
        print("model method not found: %s" % model_name)
        sys.exit(-1)
    
    # use of optimizers, see https://keras.io/optimizers/
    # according to one student, a learning rate of 0.001 yields
    # smoother turns, where 0.0001 yields sharper turns if your
    # car isn't turning far enough
    model.compile(optimizer='adam', loss='mse', lr=learning_rate)
    '''
    Setting model.fit(verbose = 1) will

        output a progress bar in the terminal as the model trains.
        output the loss metric on the training set as the model trains.
        output the loss on the training and validation sets after each epoch.
    '''
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(X_train), 
            validation_data=validation_generator, 
            nb_val_samples=len(X_val), nb_epoch=n_epochs, verbose=1)
    model.save('model.h5')
    if params["show_hist"]:
        print(history_object.history.keys())
        
        ### plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('images=%s, model=%s, drop=%05f,\nreg=%05f,lr=%05f' % 
                  (image_dir, model_name, p_dropout, p_regularize, learning_rate))
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
if __name__ == "__main__":
    main()