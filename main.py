import cv2 as cv2
import numpy as numpy
import matplotlib.pyplot as plt 
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

training_images, training_images = training_images / 255 , training_images / 255


class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

