import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras as keras
from keras import utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.optimizers import Adam
data_dir= r"C:\Users\Yasir\Downloads\Compressed\archive\Bone Break Classification\Bone Break Classification"
"""# **Creating the test and train dataset**"""
train_data=utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    validation_split=0.1,
    subset="training",
    shuffle=True,
    color_mode="rgb",
    image_size=(256,256),
    batch_size=64,
    seed=40,
)
# validation data
vald_data=utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    validation_split=0.1,
    subset="validation",
    color_mode="rgb",
    image_size=(256,256),
    batch_size=64,
    seed=40,
)
print(train_data)
# Take a single batch of images
for images,lables in train_data.take(1):
  print(images.shape)
  print(lables.shape)
print(vald_data)
for images,lables in vald_data.take(1):
  print(images.shape)
  print(lables.shape)
"""# **Pre-Process**"""
print(type(train_data))
print(type(vald_data))
classes=train_data.class_names
print(classes)
# preprocessing is an important step as it involves the normilization of images
def normalize(image, label):
  return image/255.0, label
train_data = train_data.map(normalize)
vald_data= vald_data.map(normalize)
for img, label in train_data.take(1):
  print(type(img),type(label))
"""# **Test and Train**"""
train_x=[]
train_y=[]
for image,label in train_data:
   train_x.append(image)
   train_y.append(label)
   print(type(train_y))
train_x = tf.concat(train_x, axis=0)
train_y = tf.concat(train_y, axis=0)
print(train_y)
type(train_y)
train_x
val_x=[]
val_y=[]
for image,label in train_data:
   val_x.append(image)
   val_y.append(label)
val_x = tf.concat(val_x, axis=0)
val_y = tf.concat(val_y, axis=0)
type(train_x)
type(train_y)
#Onehot Encoding
num_classes = 10
train_y = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)
val_y = tf.keras.utils.to_categorical(val_y, num_classes=num_classes)
"""# **Plot**"""
class_labels=["Avulsion fracture","Comminuted fracture","Fracture Dislocation","Greenstick fracture",
              "Hairline Fracture","Impacted fracture","Longitudinal fracture","Oblique fracture",
              "Pathological fracture","Spiral Fracture"]
# Initialize the figure and subplots
fig, axes = plt.subplots(2, 4, figsize=(15, 5))
# Iterate through the first 10 images
for i, ax in enumerate(axes.flat):
    # Select the image and label
    image, label = train_x[i], train_y[i]
    # Display the image
    ax.imshow(image, cmap='gray')
    # Set the title with the class label
    ax.set_title(f"{class_labels[np.argmax(label)]}")
    ax.axis('off')
# Display the figure
plt.show()
"""# **Model**"""
# Define the CNN model
model = Sequential()
model.add(Conv2D(60, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(120, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=10, epochs=5,
          validation_data=(val_x,val_y))
loss, accuracy = model.evaluate(val_x,val_y)
print('Test accuracy:', accuracy)
pred = model.predict(val_x)
print(pred)
num_images_to_display = 30
num_columns = 3
num_rows = (num_images_to_display + num_columns - 1)
fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 20))
for i, ax in enumerate(axes.flat):
    if i < num_images_to_display:
        ax.imshow(val_x[i])
        actual_label = class_labels[np.argmax(val_y[i])]
        predicted_label = class_labels[np.argmax(pred[i])]
        ax.set_title(f"Actual: {actual_label}, Predicted: {predicted_label}")
        ax.axis('off')
    else:
        ax.axis('off')
plt.tight_layout()
plt.show()
# After training your model
model.save('bone_fracture_model_rmspop.h5')  # Save the model as an h5 file
