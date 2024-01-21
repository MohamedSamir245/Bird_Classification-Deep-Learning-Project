import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL

from keras_preprocessing.image import ImageDataGenerator

from pathlib import Path
import random

import matplotlib.cm as cm
import cv2
import seaborn as sns

sns.set_style('darkgrid')



# Metrics
from sklearn.metrics import classification_report, confusion_matrix
import itertools

import sys
sys.path.append('src')
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir,walk_through_dir_with_logs, pred_and_plot


train_root = Path("data/external/Birds/train")
valid_root = Path("data/external/Birds/valid")
test_root = Path("data/external/Birds/test")

history, history_tuned = {}, {}

with open('src/visualization/history.json', 'r') as json_file:
    history = json.load(json_file)


with open('src/visualization/history_tuned.json', 'r') as json_file:
    history_tuned = json.load(json_file)


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy before fine tuning')
plt.legend()
plt.savefig('reports/figures/accuracy_before_fine_tuning.png')
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss before fine tuning')
plt.legend()
plt.savefig('reports/figures/loss_before_fine_tuning.png')
plt.show()


accuracy = history_tuned.history['accuracy']
val_accuracy = history_tuned.history['val_accuracy']

loss = history_tuned.history['loss']
val_loss = history_tuned.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy after fine tuning')
plt.legend()
plt.savefig('reports/figures/accuracy_after_fine_tuning.png')
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss after fine tuning')
plt.legend()
plt.savefig('reports/figures/loss_after_fine_tuning.png')
plt.show()


birda_df=pd.read_csv("data/external/Birds/birds.csv")
testData=birda_df[birda_df["data set"]=="test"]

BATCH_SIZE = 32
IMAGE_SIZE = (224,224)

test_datagen = ImageDataGenerator(rescale = 1./255)

test_data = test_datagen.flow_from_directory(directory = valid_root,
                                               batch_size = BATCH_SIZE,
                                               target_size = IMAGE_SIZE,
                                               class_mode = "categorical")

labels = (test_data.class_indices)
labels = dict((v,k) for k,v in labels.items())

model_0 = tf.keras.models.load_model('models/Birds_Classification_inceptionV3.h5')


firstdim=np.random.randint(0,82,25)
thirddim=np.random.randint(0,31,25)

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25),
                        subplot_kw={'xticks': [], 'yticks': []})


for i, ax in enumerate(axes.flat):
    ax.imshow(test_data[firstdim[i]][0][thirddim[i]])
    truelabel=np.argmax(test_data[firstdim[i]][1][thirddim[i]])
    image = tf.keras.preprocessing.image.img_to_array(test_data[firstdim[i]][0][thirddim[i]])
    image = np.expand_dims(image, axis=0)
    prediction = np.argmax(model_0.predict(image))
    if truelabel == prediction:
        color = "green"
    else:
        color = "red"
    ax.set_title(f"True: {labels[truelabel]}\nPredicted: {labels[prediction]}", color=color)

plt.savefig('reports/figures/sample_predictions.png')
plt.show()
plt.tight_layout()


