import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

BASE_PATH = os.path.join('.', 'data')
IMG_BASE_PATH = os.path.join(BASE_PATH, 'IMG')
DRIVE_LOG_PATH = os.path.join(BASE_PATH, 'driving_log.csv')

samples = []
with open(os.path.join(DRIVE_LOG_PATH)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('len(train_samples)={:d}, len(validation_samples)={:d}'.format(len(train_samples), len(validation_samples)))


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
                # choose right directory separator
                SEP = '/' if '/' in batch_sample[0] else '\\'

                # read the center image in RGB format
                name = os.path.join(IMG_BASE_PATH, batch_sample[0].split(SEP)[-1])
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                # read the steering angle
                center_angle = float(batch_sample[3])
                
                images.append(center_image)
                angles.append(center_angle)
                # data augmentation: flip image and reverse steering angle
                images.append(cv2.flip(center_image, 1))
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Activation
from keras.layers import Cropping2D, Lambda, Dropout, BatchNormalization, AveragePooling2D
from keras.optimizers import Adam


def nvidia_net():
    DROPOUT_PROB = 0.15
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(DROPOUT_PROB))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(DROPOUT_PROB))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(DROPOUT_PROB))
    model.add(Dense(1, activation='tanh'))

    return model


def mynet():
    conv_opt = {'border_mode': 'same'}
    pool_opt = {'border_mode': 'same', 'pool_size': (2, 2)}
    DROPOUT_PROB = 0.15
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # image size after cropping: (65, 320, 3)

    model.add(Convolution2D(8, 5, 5, **conv_opt))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(**pool_opt))

    model.add(Convolution2D(16, 5, 5, **conv_opt))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(**pool_opt))

    model.add(Convolution2D(24, 5, 5, **conv_opt))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(**pool_opt))

    model.add(Convolution2D(32, 3, 3, **conv_opt))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(AveragePooling2D(**pool_opt))

    model.add(Convolution2D(48, 3, 3, **conv_opt))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(200, activation='tanh'))
    model.add(Dropout(DROPOUT_PROB))
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(DROPOUT_PROB))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(DROPOUT_PROB))
    model.add(Dense(1, activation='tanh'))

    return model


optim = Adam(lr=0.001)
model = nvidia_net()
# model = mynet()
model.compile(loss='mse', optimizer=optim)

print('Training...')
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
model.fit_generator(train_generator, nb_epoch=5, samples_per_epoch=2*len(train_samples),
                    validation_data=validation_generator, nb_val_samples=2*len(validation_samples))
model.save('model.h5')
print('Model saved.')