import csv
import cv2
import numpy as np

# read all the lines of the csv
lines = []
with open('./data/joined/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

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

# images are inputs and steering angles are outputs
X_train = np.array(images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, batch_size=32)

model.save('model.h5')