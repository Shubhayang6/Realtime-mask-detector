# packages imported
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Initial learning rate defined
INIT_LR = 1e-4
# Number of epochs
EPOCHS=20
# Batch size
BS=32

# path defined
DIRECTORY = r"C:\Users\KIIT\Documents\College stuffs\Projects\Realtime mask detector\Realtime-mask-detector\dataset"
CATEGORIES = ["with_mask", "without_mask"]

print("Loading Images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image) 
        image= preprocess_input(image)

        data.append(image)
        labels.append(category)

#one hot encoding
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

data=np.array(data,dtype="float32")
labels=np.array(labels)

(trainX, testX, trainY,testY) = train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=42)

# Image generator for augmentation
aug=ImageGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# base model construction-> MobileNetV2
baseModel=MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

# head model(stays on top of the model)
headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name="flatten")(headModel)
headModel=Dense(128, activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(128, activation="softmax")(headModel)


model=Model(inputs=baseModel.input, outputs=headModel) 

# Loop over all layers in base model so that they dont get updated in first training process
for layer in baseModel.layers:
    layer.trainable=False

#model compilation
print("Compiling Model...")
opt=Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# training the head 
print("Training head...")
H=model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX)//BS,
    validation_data=(testX, testY),
    validation_steps=len(testX)//BS,
    epochs=EPOCHS)

# making prediction on testing
print("Evaulating Model...")
predIdxs=model.predict(testX, batch_size=BS)

predIdxs=np.argmax(predIdxs, axis=1)

# classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_)) 

# serializing model to disk
print("Saving Model...")
model.save(r"C:\Users\KIIT\Documents\College stuffs\Projects\Realtime mask detector\Realtime-mask-detector\mask_detector.model", save_format="h5")

# ploting the training loss and accuracy
N=EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,N),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,N),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,N),H.history["cal_accuracy"],label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(r"C:\Users\KIIT\Documents\College stuffs\Projects\Realtime mask detector\Realtime-mask-detector\plot.png")
