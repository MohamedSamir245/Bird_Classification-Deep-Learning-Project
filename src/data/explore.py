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

from functions import compute_ela_cv, convert_to_ela_image,random_sample

train_root = Path("data/external/Birds/train")
valid_root = Path("data/external/Birds/valid")
test_root = Path("data/external/Birds/test")

walk_through_dir_with_logs(train_root,"folders_logs","logs/train_data/")
walk_through_dir_with_logs(valid_root,"folders_logs","logs/valid_data/")
walk_through_dir_with_logs(test_root,"folders_logs","logs/test_data/")


birda_df=pd.read_csv("data/external/Birds/birds.csv")

print(f"Columns of the dataframe are {birda_df.columns}")
print(f"Number of images in train data = {birda_df[birda_df['data set']=='train'].shape[0]}")
print(f"Number of images in valid data = {birda_df[birda_df['data set']=='valid'].shape[0]}")
print(f"Number of images in test data = {birda_df[birda_df['data set']=='test'].shape[0]}")


trainData=birda_df[birda_df["data set"]=="train"]
validData=birda_df[birda_df["data set"]=="valid"]
testData=birda_df[birda_df["data set"]=="test"]

base = "data/external/Birds/"

trainRealPaths=[base+path for path in trainData["filepaths"]]
validRealPaths=[base+path for path in validData["filepaths"]]
testRealPaths=[base+path for path in testData["filepaths"]]

train_label_counts = trainData['labels'].value_counts()
train_label_counts_df = pd.DataFrame(train_label_counts)

# Plot the distribution of the top 20 labels in the dataset
plt.figure(figsize=(20, 6))
sns.barplot(x=train_label_counts[:20].index, y=train_label_counts[:20].values, alpha=0.8, palette=sns.color_palette("YlOrRd_r",40))
plt.title('Distribution of Top 20 Labels in Image Dataset', fontsize=16)
plt.xlabel('Label', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.savefig('reports/figures/top20_labels_distribution.png')
plt.show()

# Visualize some images
random_index = np.random.randint(0, len(trainData), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(trainRealPaths[random_index[i]]))
    ax.set_title(trainData.labels[random_index[i]],color="blue",fontsize=20)
plt.tight_layout()
plt.savefig('reports/figures/sample_images.png')
plt.show()



rnd_idx=np.random.randint(0,len(trainData))
trainRealPaths[rnd_idx].split('/')[:-1]

# Check the ELA image for different quality levels
p = random_sample('/'.join(trainRealPaths[rnd_idx].split('/')[:-1]))
orig = cv2.imread(p)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 255.0
init_val = 100
columns = 3
rows = 3

fig=plt.figure(figsize=(15, 10))
for i in range(1, columns*rows +1):
    quality=init_val - (i-1) * 8
    img = compute_ela_cv(path=p, quality=quality)
    if i == 1:
        img = orig.copy()
    ax = fig.add_subplot(rows, columns, i) 
    ax.title.set_text(f'q: {quality}')
    plt.imshow(img)
plt.savefig('reports/figures/ela_images.png')
plt.show()