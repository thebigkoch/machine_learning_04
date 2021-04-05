# See https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c02_dogs_vs_cats_with_augmentation.ipynb

import tensorflow as tf

# Used for working with data on disk to interface with our model.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import pathlib

# Constants
EPOCHS = 100
BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
IMG_SHAPE  = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

def create_rgb_image_model() -> tf.keras.models.Sequential :
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        ############################
        # CHANGED HERE
        ############################
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    return model

def compile_rgb_image_model(model: tf.keras.models.Sequential) -> tf.keras.models.Sequential :
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def visualize_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./foo.png')
    plt.show()

class MachineLearning04:
    def __init__(self):
        self._URL = ''
        self.zip_dir = None
        self.base_dir = None
        self.train_dir = None
        self.validation_dir = None

        self.train_cats_dir = None
        self.train_dogs_dir = None
        self.validation_cats_dir = None
        self.validation_dogs_dir = None

        self.num_cats_tr = None
        self.num_dogs_tr = None

        self.num_cats_val = None
        self.num_dogs_val = None

        self.total_train = None
        self.total_val = None

        self.train_image_generator      = None
        self.validation_image_generator = None

        self.train_data_gen = None
        self.val_data_gen = None

        self.model = None
        
    def load_data(self):
        current_directory = os.path.dirname(os.path.realpath(__file__)) 
        desired_file = os.path.join(current_directory, 'data', 'cats_and_dogs_filtered.zip')
        print(desired_file) 
        if os.path.exists(desired_file):
            print('Using local copy of cats & dogs data.')
            self._URL = pathlib.Path(desired_file).as_uri()
        else:
            print('Using remote copy of cats & dogs data.')
            self._URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
        self.zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=self._URL, extract=True)

        self.base_dir = os.path.join(os.path.dirname(self.zip_dir), 'cats_and_dogs_filtered')
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.validation_dir = os.path.join(self.base_dir, 'validation')

        self.train_cats_dir = os.path.join(self.train_dir, 'cats')  # directory with our training cat pictures
        self.train_dogs_dir = os.path.join(self.train_dir, 'dogs')  # directory with our training dog pictures
        self.validation_cats_dir = os.path.join(self.validation_dir, 'cats')  # directory with our validation cat pictures
        self.validation_dogs_dir = os.path.join(self.validation_dir, 'dogs')  # directory with our validation dog pictures

        self.num_cats_tr = len(os.listdir(self.train_cats_dir))
        self.num_dogs_tr = len(os.listdir(self.train_dogs_dir))

        self.num_cats_val = len(os.listdir(self.validation_cats_dir))
        self.num_dogs_val = len(os.listdir(self.validation_dogs_dir))

        self.total_train = self.num_cats_tr + self.num_dogs_tr
        self.total_val = self.num_cats_val + self.num_dogs_val

    def print_metadata(self):
        print('total training cat images:', self.num_cats_tr)
        print('total training dog images:', self.num_dogs_tr)

        print('total validation cat images:', self.num_cats_val)
        print('total validation dog images:', self.num_dogs_val)
        print("--")
        print("Total training images:", self.total_train)
        print("Total validation images:", self.total_val)

    """
    Images must be formatted into appropriately pre-processed floating point tensors before being fed into the network. The steps involved in preparing these images are:
    1) Read images from the disk
    2) Decode contents of these images and convert it into proper grid format as per their RGB content
    3) Convert them into floating point tensors
    4) Rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values.

    All these tasks can be completed with ImageDataGenerator.
    """
    def load_image_data_generators(self):

        ############################
        # CHANGED HERE
        ############################
        self.train_image_generator = ImageDataGenerator(rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')  # Generator for our training data

        self.validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

        # We can read the data from disk, rescale, and resize with flow_from_directory().
        self.train_data_gen = self.train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=self.train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

        self.val_data_gen = self.validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=self.validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE), #(150,150)
                                                              class_mode='binary')

    # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    def plotImages(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()

    def visualize_training_images(self):
        sample_training_images, _ = next(self.train_data_gen) 
        self.plotImages(sample_training_images[:5])  # Plot images 0-4

    def create_and_train_model(self):
        self.model = create_rgb_image_model()
        self.model = compile_rgb_image_model(self.model)

        # Print the model summary.
        print("Summary of RGB Image Model")
        print("--------------------------")
        self.model.summary()

        history = self.model.fit_generator(
            self.train_data_gen,
            steps_per_epoch=int(np.ceil(self.total_train / float(BATCH_SIZE))),
            epochs=EPOCHS,
            validation_data=self.val_data_gen,
            validation_steps=int(np.ceil(self.total_val / float(BATCH_SIZE)))
        )

        return history

def main():
    ml04 = MachineLearning04()
    ml04.load_data()
    ml04.print_metadata()
    ml04.load_image_data_generators()

    # Uncomment to see some sample training images.
    # ml04.visualize_training_images()

    history = ml04.create_and_train_model()
    visualize_history(history)

if __name__ == "__main__":
    main()
