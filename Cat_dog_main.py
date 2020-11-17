import matplotlib.pyplot as plt
import warnings
from matplotlib import gridspec
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

#For distribution of images run Categorisation first

train_dir = "./Data/train"
validation_dir = "./Data/validation"

# Reproducability
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells


#Data preprosessing

# rescaling all images by 255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Resizing all images to 150 X 150 and using binary_crossentropy loss because we need binary labels
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                    batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), 
                                                        batch_size=20, class_mode='binary')

# checking the output of one of the generators
for data_batch, labels_batch in train_generator:
    print('data batch shape: ', data_batch.shape)
    print('labels batch shape: ', labels_batch.shape)
    break


#Adding a pretrained base
pretrained_base = tf.keras.applications.InceptionV3()

pretrained_base.trainable = False

#Creating the model
model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(units=6, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
    ])

#Choosing an optimizer for the model
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)

#compiling the model
model.compile(
    optimizer=optimizer,
    loss = 'binary_crossentropy',
    metrics=['binary_accuracy']
    )



history = model.fit_generator(
    train_generator, 
    steps_per_epoch=100,
    epoch=100,
    epochs=30
    validaiton_data=validation_generator,
    validation_steps=50
    )


import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();








