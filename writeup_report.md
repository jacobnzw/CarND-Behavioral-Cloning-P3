# **Behavioral Cloning** 

---

**Project Goals**

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

---
### Project Files

My project includes the following files:
* `model.py` containing code for loading the training data and code for buidling and training the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` project writeup summarizing the results and methods used

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The `model.py` file contains specification of the convolutional neural network architecture and the code for training and saving of the final trained model. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Collecting Data

** *TODO: how data was collected? Maybe provided dataset is enough? Feels better with my own dataset ;)* **

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 


To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.



### Model Architecture
---
*The overall strategy for deriving a model architecture was to ...*

*My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...*

*In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. *

*The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....*

*At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.*

---

My model is heavily inspired by [this](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) article from NVidia Developer Blog authored by Mariusz Bojarski, Ben Firner, Beat Flepp, Larry Jackel, Urs Muller, Karol Zieba and Davide Del Testa. 

As a pre-processing step, I'm normalizing the data first using the Keras' `Lambda` layer, like so
```python
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
```
and then I crop the image using the `Cropping2D` layer, like so
```python
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
```
in order to restrict model's attention on the road and ignore the distracting aspects of the image (like the sky and the track surroundings). The layer crops 70 pixels from the top and 25 pixels from the bottom, leaving us with a 65x320x3 image.

The model is a convolution neural network in which the first three convolutional layers use 5x5 filters with 24, 36 and 48 filters respectively. (`model.py` lines 64-66) Each of the three layers is using step size of 2. The following two convolutional layers use 64 of 3x3 filters with step size 1. All convolutional layers use the `'valid'` border mode and RELU as an activation function.

The outputs of the feature extraction part of the network are flattened and fed into the classifier, which consists of three fully-connected layers with progressively decreasing number of neurons (100, 50, 10 and 1). Looking through the dataset I saw that the steering angle is represented as a number ranging from -1 to 1. For this reason, I decided to use the hyperbolic tanget activation function (instead of RELU) for all fully-connected layers.

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
to allow me to control the learning rate and other parameters more closely. As evident from the code snippet above and after some experimentation, I eventually settled on the learning rate of 0.001. In order to cope with the large data set, I created generators for the training and validation sets respectively and trained using the `fit_generator()` function. I trained for 5 epochs with the batch size of 32.
```python
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
model.fit_generator(train_generator, nb_epoch=5, samples_per_epoch=2*len(train_samples),
                    validation_data=validation_generator, nb_val_samples=2*len(validation_samples))
```

The model was trained and validated on different data sets to ensure that the model was not overfitting. One fifth (20%) of the available data was reserved for validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.



#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]
