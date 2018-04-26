import csv
import cv2
import numpy as np

# read all the lines of the csv
lines = []
with open('./data/data_forw_mouse_2laps/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
steering_angles = []
for line in lines:
    # get the image path
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './data/data_forw_mouse_2laps/IMG/' + filename

    # load the image using OpenCV
    image = cv2.imread(current_path)
    images.append(image)

    # steering angle stored in the 4th column of the driving_log.csv
    steering_angles.append(float(line[3]))

# images are inputs and steering angles are outputs
X_train = np.array(images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D

# TODO: data normalization using Lambda layers (might glitch)
# TODO: BatchNormalization for deep nets
# TODO: Transfer learning?

conv_opt = {'border_mode': 'same', 'activation': 'relu'}
model = Sequential()
model.add(Convolution2D(4, 5, 5, input_shape=(160, 320, 3), **conv_opt))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Convolution2D(8, 5, 5, **conv_opt))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 3, 3, **conv_opt))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
model.add(Convolution2D(32, 3, 3, **conv_opt))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, batch_size=128)

model.save('model.h5')