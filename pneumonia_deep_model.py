# %%
import os
import numpy as np
import matplotlib.pyplot as plt

base_dir = '/home/hk633839/Downloads/dataset/chest_xray/'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')

# variables for counting the number of images for each class
pos = 0
neg = 0
train_normal = os.path.join(train_dir, 'NORMAL')
train_disease = os.path.join(train_dir, 'PNEUMONIA')
for path in os.listdir(train_normal):
    if os.path.isfile(os.path.join(train_normal, path)):
        pos += 1
for path in os.listdir(train_disease):
    if os.path.isfile(os.path.join(train_disease, path)):
        neg += 1

test_normal = os.path.join(test_dir, 'NORMAL')
test_disease = os.path.join(test_dir, 'PNEUMONIA')
for path in os.listdir(test_normal):
    if os.path.isfile(os.path.join(test_normal, path)):
        pos += 1
for path in os.listdir(test_disease):
    if os.path.isfile(os.path.join(test_disease, path)):
        neg += 1

validation_normal = os.path.join(val_dir, 'NORMAL')
validation_disease = os.path.join(val_dir, 'PNEUMONIA')
for path in os.listdir(validation_normal):
    if os.path.isfile(os.path.join(validation_normal, path)):
        pos += 1
for path in os.listdir(validation_disease):
    if os.path.isfile(os.path.join(validation_disease, path)):
        neg += 1
print(pos, neg)

# %%
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

def dataGen(img_size, batch_size, seed):

    train_datagen = ImageDataGenerator(rescale = 1./255,rotation_range = 15,width_shift_range = 0.1,height_shift_range = 0.1,shear_range = 0.1,zoom_range = 0.1,
                                       brightness_range = (0.95, 1.05),vertical_flip = False,horizontal_flip = False,fill_mode = 'nearest')
    test_datagen = ImageDataGenerator(rescale = 1./255)
    val_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,target_size = (img_size, img_size),batch_size = batch_size,class_mode = 'binary',shuffle = True,seed = seed)
    
    test_generator = test_datagen.flow_from_directory(test_dir,target_size = (img_size, img_size),batch_size = batch_size,class_mode = 'binary',shuffle = True,seed = seed)
    
    val_generator = val_datagen.flow_from_directory(val_dir,target_size = (img_size, img_size),batch_size = batch_size,class_mode = 'binary',shuffle = True,seed = seed)
    
    class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(train_generator.classes), y = train_generator.classes)
    class_weights = dict(zip(np.unique(train_generator.classes), class_weights))
    return train_generator, test_generator, val_generator, class_weights

# %%
def plot_acc_loss():
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    precision = history.history['precision']
    recall = history.history['recall']
    epochs_range = range(len(accuracy))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, 'b', label='Validation Accuracy')
    plt.legend(loc='best')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, 'r', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'b', label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, precision, 'r', label='Precision')
    plt.plot(epochs_range, recall, 'b', label='Recall')
    plt.legend(loc='best')
    plt.title('Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall vs Precision')

    plt.tight_layout()

    plt.show()

# %%
import math
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Callback functions
def decay(lr):
    def decay_fn(epoch):
        return lr * 0.1 ** (epoch / 15)
    return decay_fn
decay_fn = decay(0.02)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay_fn)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)

# %%
IMAGE_SIZE = [299, 299]
batch_size = 128
seed = 42

# Preprocessing
train_generator, test_generator, val_generator, class_weights = dataGen(IMAGE_SIZE[0], batch_size, seed)

# %%
initial_bias = np.log([pos/neg])
initial_bias

# %%
def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
    
    return block

# %%
def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block

# %%
model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),

        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.summary()

# %%
# Set the training parameters
METRICS = ['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = METRICS)

EPOCHS = 100
# Train the model
history = model.fit(train_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=train_generator.samples//train_generator.batch_size,
                    validation_data=test_generator,
                    validation_steps=test_generator.samples//test_generator.batch_size,
                    class_weight=class_weights,
                    callbacks=[lr_scheduler, early_stopping],
                    verbose=1)
model.save('pneumonia.keras')

# %%
plot_acc_loss()


