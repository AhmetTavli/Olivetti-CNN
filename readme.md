[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://github.com/AhmetTavli/Badge/blob/master/badges/pytorch-badge.svg)](https://pytorch.org/)
[![Keras](https://github.com/AhmetTavli/Badge/blob/master/badges/keras_badge.svg)](https://keras.io/)
[![Python](https://upload.wikimedia.org/wikipedia/commons/f/fc/Blue_Python_3.7_Shield_Badge.svg)](https://www.python.org/)
[![Numpy](https://github.com/AhmetTavli/Badge/blob/master/badges/numpy_badge.svg)](https://numpy.org/)
[![PyCharm](https://github.com/AhmetTavli/Badge/blob/master/badges/pycharm_badge.svg)](https://www.jetbrains.com/pycharm/)

# Olivetti - CNN

To run:

`python -W ignore olivetti.py --write-to-file True --generate-data True`

 The test-set accuracy should be close to the 97%.
 
 ## Steps
 
### 1. The dataset is divided into the 65% - 35% ratio.

We have 400 samples; 260 training images and 140 test images.

If we train the model on 260 images the maximum accuracy will be 5%.

`python -W ignore olivetti.py --write-to-file True`

### 2- Generate images 

We will have two types of image generator: train and test

```python
train_data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
```

So what does it mean?

From a selected sample:

![Image of example_img](3_0.png)

10 samples generated based on the shearing, zoom and horizontal flip:

![Image of train_gen](gen_train_samples.png)
   
### 3. Train and Test Loss (After 5 epoch)

 ![Image of loss](olivetti_loss.png)


## Contributing :thought_balloon:
Pull requests are welcome.

For major changes, please open an issue, then discuss what you would like to change.

 ## License :scroll:
[MIT](https://opensource.org/licenses/MIT)
