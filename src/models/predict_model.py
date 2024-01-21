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

model_0 = tf.keras.models.load_model('models/Birds_Classification_InceptionV3.h5')


y_test=[]
pred=[]

cur=1

for _ in range(2625):
    img, label = test_data.next()
    y_test.append(np.argmax(label))
    pred.append(np.argmax(model_0.predict(img)))
    print(cur)
    cur+=1

report = classification_report(y_test, pred, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv('reports/classification_report.csv', index=False)