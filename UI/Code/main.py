import tkinter
import customtkinter
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
import cv2

import json


IMAGE_PATH = "UI/Code/OIP.jpg"
LABELS_DICT = {}
MODEL_DETAILS = {}


FINAL_PREDS = []
FINAL_PROBAS = []


def init():

    global LABELS_DICT
    global MODEL_DETAILS

    with open("UI/Code/labels.json", 'r') as f:
        LABELS_DICT = json.load(f)

    with open("UI/Code/modelDetails.json", 'r') as f:
        MODEL_DETAILS = json.load(f)

    model = tf.keras.models.load_model(
        "UI/Code/InceptionV3_finetuned_input224_earlystopping.h5")
    return model


model = init()


def imageChange_callback():
    img2 = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH),
                                  size=(450, 450))

    image_label.configure(image=img2)


def upload_file():
    file_path = customtkinter.filedialog.askopenfilename(title="Select an image", filetypes=[(
        'all files', '*.*'), ('png files', '*.png'), ('jpeg files', '*.jpeg'), ('jpg files', '*.jpg')])

    global IMAGE_PATH
    IMAGE_PATH = file_path
    imageChange_callback()


def preprocess_image():
    orig_img = cv2.imread(IMAGE_PATH)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    orig_img = cv2.resize(orig_img, (224, 224))
    orig_img = orig_img/255

    image = np.expand_dims(orig_img, axis=0)

    return image


def predict():
    img = preprocess_image()
    top_5_predictions = tf.nn.top_k(model.predict(img), k=5)

    # Sort the predictions by probability in descending order.
    probas = top_5_predictions[0].numpy()[0]
    preds = top_5_predictions[1].numpy()[0]
    probas = probas*100
    probas = ["{:.9f}".format(p) for p in probas]

    probas_list = []
    for x, y in np.nditer([preds, probas]):
        dictionary = {int(x): str(y)}
        probas_list.append(dictionary)

    # Create a function that returns the probability of the dictionary.
    def get_probability(dictionary):
        return dictionary[list(dictionary.keys())[0]]

    # Sort the list of dictionaries using the function as the key.
    sorted_list_of_dictionaries = sorted(
        probas_list, key=get_probability, reverse=True)

    final_preds = []
    final_probas = []

    for dict in sorted_list_of_dictionaries:
        final_preds.append(LABELS_DICT[str(next(iter(dict)))])
        final_probas.append(next(iter(dict.values())))

    global FINAL_PREDS
    FINAL_PREDS = final_preds

    global FINAL_PROBAS
    FINAL_PROBAS = final_probas

    updatePredictionLabels()


def updatePredictionLabels():
    for i in range(5):
        predictionLabelArray[i].configure(
            text=FINAL_PREDS[i]+"   "+FINAL_PROBAS[i]+"%\n\n-------------------------------------------------------------------")
        predictionLabelArray[i].update()


def toggle_prediction_frame():
    global frame_enabled
    if not frame_enabled:
        scrollbarFrame.pack(fill='both', expand=True)
        modelDetailsFrame.pack_forget()
        birdsButton.configure(text="Show model details")
    else:
        scrollbarFrame.pack_forget()
        modelDetailsFrame.pack(fill='both', expand=True)
        birdsButton.configure(text="Show birds list")
    frame_enabled = not frame_enabled


customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("1080x720")

app.title("Bird Classification")

title = customtkinter.CTkLabel(app, text="Predict the type of bird from an image with over 525 possible species",
                               font=customtkinter.CTkFont(family='Arial', size=20, weight='bold'))
title.pack(padx=10, pady=(40, 80))

mainFrame = customtkinter.CTkFrame(app, fg_color="transparent")
mainFrame.pack(fill="x", expand=True, anchor="n")

my_image = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH),
                                  size=(450, 450))


predictionsFrame = customtkinter.CTkFrame(
    mainFrame, width=340, height=400)
predictionsFrame.pack_propagate(False)
predictionsFrame.pack(side="right", padx=(0, 10), anchor="n")

predictionTitle = customtkinter.CTkLabel(
    predictionsFrame, text="Top-5 Accuracy", font=customtkinter.CTkFont(family='Arial', size=20, weight='bold'))
predictionTitle.pack(pady=(5, 10))

predictionLabelArray = []
for i in range(5):
    predictionLabel = customtkinter.CTkLabel(
        predictionsFrame, text=f"--- Predection {i} ---\n\n-------------------------------------------------------------------", font=('Arial', 13))
    predictionLabel.pack(pady=12)
    predictionLabelArray.append(predictionLabel)

frame_enabled = False

imageButtonsFrame = customtkinter.CTkFrame(
    mainFrame, width=450, height=500, fg_color="transparent")
imageButtonsFrame.pack_propagate(False)
imageButtonsFrame.pack(side="right", padx=(10))

image_label = customtkinter.CTkLabel(
    imageButtonsFrame, image=my_image, text="")  # display image with a CTkLabel
image_label.pack()

buttonsFrame = customtkinter.CTkFrame(
    imageButtonsFrame, fg_color="transparent")
buttonsFrame.pack(pady=10, anchor="n")

uploadButton = customtkinter.CTkButton(
    buttonsFrame, text="Upload an image", command=upload_file)
uploadButton.pack(side="left", padx=(5, 5))

predictButton = customtkinter.CTkButton(
    buttonsFrame, text="Predict", command=predict)
predictButton.pack(side="right", padx=(5, 5))

birdsButton = customtkinter.CTkButton(
    buttonsFrame, text="Show birds list", command=toggle_prediction_frame)
birdsButton.pack(padx=(5, 5))


my_frame = customtkinter.CTkFrame(mainFrame)
my_frame.pack(fill='both', expand=True, padx=(10, 0))


scrollbarFrame = customtkinter.CTkScrollableFrame(my_frame)
# scrollbarFrame.pack(fill='both', expand=True)

modelDetailsFrame = customtkinter.CTkScrollableFrame(my_frame)
modelDetailsFrame.pack(fill="both", expand=True)
modelLabel = customtkinter.CTkLabel(
    modelDetailsFrame, text="Model Details:", justify="center")
modelLabel.pack(anchor="w")

for key, value in MODEL_DETAILS.items():
    mLabel = customtkinter.CTkLabel(
        modelDetailsFrame, text=f"{key}. {value}\n", wraplength=200, justify="left")
    mLabel.pack(anchor="w")


for key, value in LABELS_DICT.items():
    birdLabel = customtkinter.CTkLabel(scrollbarFrame, text=f"{key}. {value}")
    birdLabel.pack(anchor="w")


app.mainloop()
