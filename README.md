# machine_learning_04

Udacity course: https://classroom.udacity.com/courses/ud187

## Lesson 5: Going Further With CNNs

### Overfitting
Overfitting can be a common problem with CNNs.

* Caused by the **bias-variance tradeoff**.
* The *bias error* is an error from false assumptions, often caused by not having enough weights.
* The *variance error* is an error from too much sensitivity to small fluctuations (variances) in the training set. 

Possible solutions:
* Holdout sets: Check your performance on the test datasets.  As soon as error starts to *increase* again, you know you're overfitting, so you stop training.
* Constrain your network: Put biases on specific weights.  The more you constrain your weights, the less chance of overfitting.
* Training / validation / test datasets: **Only** use the test dataset for testing your results.

### Working with high-res color images of different sizes
Going to work with Microsoft Asirra dataset, which contains pictures of cats and dogs.

#### Working with images of different sizes
CNNs *must* operate on inputs of the same size.  For this reason, the images of different sizes must be resizde to the same 2D size prior to flattening.

#### Working with color images
This works the same as 2D grayscale images that have a width and height, but now there is also a "depth".  The depth is determined by the number of color channels.  So this will typically be the three color channels -- Red, Green, and Blue:

![](images/01.png)

#### CNNs with color images
Much like with 2D images, 3D RGB images also utilize convolutional layers and max pooling layers.

#### Convolutional layers for color images
The `input_shape` is now 3D.  Similarly, the kernel is also now 3D.  In fact, it's customary to use multiple 3D kernels when dealing with a color image:

![](images/03.png)

With the `keras` API, we still use the Conv2D interface, but our parameters change slightly. In 2D we had:

```
# Fashion MNIST
model = Sequentional()
model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(28,28,1)))
```

In 3D, we now have:
```
# Dogs and Cats
model = Sequentional()
model.add(Conv2D(3, (3,3), padding='same', activation='relu', input_shape=(28,28,3)))
```
**Note**: The `filters` parameter is now only `3`.  And the `kernel_size` parameter is `(3,3)`.  This gives us the three 3D kernels that we desire.

#### Max pooling layers for color images
Just like with a grayscale image, we also have a sliding window with a stride length between them.  In this example, we use a 2x2 sliding window with a stride length of 2:

![](images/04.png)

The end result is an image that has a width and height that are half the size, but with the same depth.

### Colab Example
The Colab example is available here: https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c01_dogs_vs_cats_without_augmentation.ipynb

### Next steps

## Sample Code

### System Requirements

* Python 3.7
* Pip 20.1+
* Poetry 1.1.4+:
  * `pip install poetry`
* (*On Windows 7 or later*) [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads)
  * Required by tensorflow.  [Details](https://www.tensorflow.org/install/pip#system-requirements).

### Project Setup
To setup the virtual environment, run:
  > `poetry install`

To execute the sample code, run:
  > `poetry run main`
