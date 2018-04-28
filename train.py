# import csv
# import cv2
# import numpy as np
#
# # read all the lines of the csv
# print('Reading CSV...')
# lines = []
# with open('./data/joined/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)
#
# print('Creating NumPy arrays...')
# images = []
# steering_angles = []
# for line in lines:
#     # get the image path
#     source_path = line[0]
#     filename = source_path.split('\\')[-1]
#     current_path = './data/joined/IMG/' + filename
#
#     # load the image using OpenCV
#     image = cv2.imread(current_path)
#     images.append(image)
#     # steering angle stored in the 4th column of the driving_log.csv
#     steering_angles.append(float(line[3]))
#
#     # augment training set with horizontally mirrored images
#     images.append(cv2.flip(image, 1))
#     # reverse steering angle for mirrored images
#     steering_angles.append(-float(line[3]))
#
# # images are inputs and steering angles are outputs
# X_train = np.array(images)
# y_train = np.array(steering_angles)

import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

BASE_PATH = './mydata/joined/'
samples = []
with open(BASE_PATH + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # for all CSV lines in the batch
            for batch_sample in batch_samples:
                # read the center image and steering angle
                name = BASE_PATH + 'IMG/' + batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


ch, row, col = 3, 80, 320  # Trimmed image format


from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Cropping2D, Lambda, Dropout

# TODO: Data augmentation: cv2.flip(im, 1), measurements
# TODO: Use additional image adata from left/right cameras
# TODO: BatchNormalization for deep nets
# TODO: Transfer learning?


def nvidia_net():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
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
    DROPOUT_PROB = 0.15
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # image size after cropping: (65, 320, 3)
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
    model.add(Dropout(DROPOUT_PROB))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(DROPOUT_PROB))
    model.add(Dense(1, activation='relu'))

    return model


model = mynet()
model.compile(loss='mse', optimizer='adam')

print('Training...')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, batch_size=128)
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)
model.fit_generator(train_generator, nb_epoch=3, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples))
model.save('model.h5')
print('Model saved.')