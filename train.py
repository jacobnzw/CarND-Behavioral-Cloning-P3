import csv
import cv2
import numpy as np

# read all the lines of the csv
print('Reading CSV...')
lines = []
with open('./data/joined/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print('Creating NumPy arrays...')
images = []
steering_angles = []
for line in lines:
    # get the image path
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './data/joined/IMG/' + filename

    # load the image using OpenCV
    image = cv2.imread(current_path)
    images.append(image)
    # steering angle stored in the 4th column of the driving_log.csv
    steering_angles.append(float(line[3]))

    # augment training set with horizontally mirrored images
    images.append(cv2.flip(image, 1))
    # reverse steering angle for mirrored images
    steering_angles.append(-float(line[3]))

# images are inputs and steering angles are outputs
X_train = np.array(images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Cropping2D, Lambda

# TODO: Data augmentation: cv2.flip(im, 1), measurements
# TODO: Use additional image adata from left/right cameras
# TODO: data normalization using Lambda layers (might glitch)
# TODO: Cropping2D
# TODO: BatchNormalization for deep nets
# TODO: Transfer learning?

def nvidia_net():
    model = Sequential()
    # TODO: add normalization
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))

    return model

def mynet():
    conv_opt = {'border_mode': 'same', 'activation': 'relu'}
    pool_opt = {'border_mode': 'same', 'pool_size': (2, 2)}
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Convolution2D(4, 5, 5, **conv_opt))
    # model.add(MaxPooling2D(**pool_opt))
    model.add(Convolution2D(8, 5, 5, **conv_opt))
    model.add(MaxPooling2D(**pool_opt))
    model.add(Convolution2D(16, 3, 3, **conv_opt))
    # model.add(MaxPooling2D(**pool_opt))
    model.add(Convolution2D(32, 3, 3, **conv_opt))
    model.add(MaxPooling2D(**pool_opt))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1, activation='relu'))

    return model

model = mynet()
model.compile(loss='mse', optimizer='adam')

print('Training...')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, batch_size=128)

model.save('model.h5')
print('Model saved.')