# **Behavioral Cloning** 

---

**Project Goals**

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center_driving_track2]: ./examples/center_driving_track2.jpg "Track 2 center driving."

---
### Project Files
My project includes the following files:
* `model.py` containing code for loading the training data and code for buidling and training the model
* `drive.py` for driving the car in autonomous mode
* `model_nvidia_5ep_tr1.h5` model driving on the Track 1
* `model_nvidia_6ep_tr2.h5` model driving on the Track 2
* `run_nvidianet.mp4` video recording the driving behavior of the model on Track 1
* `run_tr2.mp4` video recording the driving behavior of the model on Track 2
* `writeup_report.md` project writeup summarizing the results and methods used

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The `model.py` file contains specification of the convolutional neural network architecture and the code for training and saving of the final trained model. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Collecting Data
As a data set I used the provided Udacity data from track 1. 

In later stages of the project, I decided to give the track 2 a try. Upon realizing that track 2 is contrived and unrealistic, I proceeded to collect 4 laps of center driving using a mouse as a controller as it provided smoother angle transitions. 

![Typical image of center driving on Track 2.][center_driving_track2]

Naturally, I did not manage to keep the car on the center line for the entirety of the track so minor deviations are also present in the data set. The whole data set including the images from the left and right cameras has about 25k+ samples.


### Model Architecture
As a pre-processing step, I'm normalizing the data first using the Keras' `Lambda` layer, like so
```python
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
```
and then I crop the image using the `Cropping2D` layer, like so
```python
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
```
in order to restrict model's attention on the road and ignore the distracting aspects of the image (like the sky and the track surroundings). The layer crops 70 pixels from the top and 25 pixels from the bottom, leaving us with a 65x320x3 image.

I ended up using a model, which is heavily inspired by [this](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) article from NVidia Developer Blog authored by Mariusz Bojarski, Ben Firner, Beat Flepp, Larry Jackel, Urs Muller, Karol Zieba and Davide Del Testa. The model is a convolutional neural network in which the first three convolutional layers use 5x5 filters with 24, 36 and 48 filters respectively. Each of the three layers is using step size of 2. The following two convolutional layers use 64 of 3x3 filters with step size 1. All convolutional layers use the `'valid'` border mode and RELU as an activation function.

The outputs of the feature extraction part of the network are flattened and fed into the classification part, which consists of three fully-connected layers with progressively decreasing number of neurons (100, 50, 10 and 1). Looking through the dataset I saw that the steering angle is represented as a number ranging from -1 to 1. For this reason, I decided to use the hyperbolic tanget activation function (instead of RELU) for all fully-connected layers.

During my experiments with training different models, I observed that for large enough number of epochs the training error drops below the validation error, which indicates that the model is overfitting the training set. To prevent this, I interleaved the final fully-connected layers with the dropout layers. The dropout probability was set to `DROPOUT_PROB = 0.15`.

The following code snippet summarizes the final model architecture as specified using the Keras library.
```python
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
```


### Training Strategy
I used the ADAM (adaptive moment estimation) optimizer for training, which was instantiated explicitly using
```python
optim = Adam(lr=0.001)
```
to allow me to control the learning rate and other parameters more closely. I used the mean squared error (MSE) as a loss function. As evident from the code snippet above and after some experimentation, I eventually settled on the learning rate of 0.001. In order to cope with the large data set, I created generators for the training and validation sets respectively and trained using the `fit_generator()` function. I trained for 5 epochs with the batch size of 32.
```python
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
model.fit_generator(train_generator, nb_epoch=5, samples_per_epoch=2*len(train_samples),
                    validation_data=validation_generator, nb_val_samples=2*len(validation_samples))
```

The model was trained and validated on different data sets to ensure that the model was not overfitting. One fifth (20%) of the available data was reserved for validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


### Track 2 Driving
I was curious to see whether the NVidia model is capable of learning the contrivances of the second track. I used my own data set as described in the above section "Collecting Data". I used the same training strategy and the same batch size. 

At first, I though I might reduce the amount by which the vertical dimension of the image is cropped. The retionale behind this stems from the idea that as the car goes downhill the field of view of the camera is largely occupied by the road. I realized however, that this works against me when the car goes uphill, in which case the image is more likely to contain distracting elements. Thus I decided to leavy the settings for cropping unchanged.

I soon noticed that the model quickly overfits the training data, which was evident from the fact that the training error was lower than the validation error. This meant I had to decrease the model complexity by incresing the dropout probability to 0.25. After trainig for 6 epochs the model is able to drive autonomously on the Track 2. 


### Future Work
In the future, it would be interesting to find out whether the NVidia model can drive both tracks.