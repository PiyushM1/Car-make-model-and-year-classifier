# Car make model and year classifier
This notebook trains three separate models to identify the make, model and year of a given car. They are trained using the [Cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), which contains 16,185 images of 196 classes of cars. The classes include 49 different labels for the make, 174 different labels for the model and 16 different labels for the year of production.

The training is done using [Monk](https://github.com/Tessellate-Imaging/monk_v1) library, which is a low code Deep Learning tool and a unified wrapper for Computer Vision.

The commands to install the required libraries and to download and pre-process the dataset are included in the notebook, along with the descriptions wherever required.

For training, different models (resnet50, inception, both trained from scratch as well as using transfer learning), learning rates (from 0.001 to 0.1 in a number of steps), optimizers (stochastic gradient descent and RMSProp), batch sizes (16, 32, 64 and 128) and input sizes, were experimented with, and finally inception_v3 was used with transfer learning while freezing some initial layers during training, along with stochastic gradient descent, with the other hyper-parameters more fine tuned to each specific model.

The performance can be further improved by training for some more time, by using learning rate decay, using different optimizer, or even using a better model architecture.

The workspace with the trained models, intermediate models and training logs can be found [here](https://drive.google.com/drive/folders/13BeQqmqzZYHTrLfer_tPswU79sYJEcjN?usp=sharing).

I have also developed a Flask API to return the predictions for any uploaded image. Check it out [here](https://github.com/PiyushM1/Car-classification-API).
