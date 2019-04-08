# Histopathology Cancer Detection 
# DenseNet201 CNN classification

import numpy as np
import pandas as pd 
import os
from glob import glob
from random import shuffle
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Lambda, Multiply, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras.models import Model
from keras.applications.densenet import DenseNet201
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# import data
print(os.listdir("D:/Machine-Learning-Resource/Demo/Histopathology"))

df_train = pd.read_csv("train_labels.csv")
id_label_map = {k:v for k,v in zip(df_train.id.values, df_train.label.values)}
df_train.head()

def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tif', '')
    
    
labeled_files = glob('D:/Machine-Learning-Resource/Demo/Histopathology/train/*.tif')
test_files = glob('D:/Machine-Learning-Resource/Demo/Histopathology/test/*.tif')

print("labeled_files size :", len(labeled_files))
print("test_files size :", len(test_files))

# partition data
train, val = train_test_split(labeled_files, test_size=0.075, train_size=0.15, random_state=101010) #use fraction of data to avoid crashing

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# image augmentation
def data_gen(list_files, id_label_map, batch_size, augment=True):
    keras_gen = ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    zoom_range=0.2,
                    shear_range=5)
    while True:
        shuffle(list_files)
        for batch in chunker(list_files, batch_size):
            X = [cv2.imread(x) for x in batch]
            Y = [id_label_map[get_id_from_file_path(x)] for x in batch]
            if augment:
                X = [keras_gen.random_transform(x) for x in X]
            X = [preprocess_input(x.astype(np.float32)) for x in X]
                
            yield np.array(X), np.array(Y)
    
 # define densenet model    
def get_model_densenet():
    inputs = Input((96, 96, 3))
    base_model = DenseNet201(include_top=False, input_shape=(96, 96, 3))#, weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['acc'])
    model.summary()

    return model  

model = get_model_densenet()

# model parameters
batch_size=32
h5_path = "DenseNet_model.h5"
checkpoint = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(
    data_gen(train, id_label_map, batch_size, augment=True),
    validation_data=data_gen(val, id_label_map, batch_size),
    epochs=10, verbose=1,
    callbacks=[checkpoint],
    steps_per_epoch=len(train) // batch_size,
    validation_steps=len(val) // batch_size) 

preds = []
ids = []

for batch in chunker(test_files, batch_size):
    X = [preprocess_input(cv2.imread(x).astype(np.float32)) for x in batch]
    ids_batch = [get_id_from_file_path(x) for x in batch]
    X = np.array(X)
    preds_batch = ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
    preds += preds_batch
    ids += ids_batch
    
df = pd.DataFrame({'id':ids, 'label':preds})
df.to_csv("densenet201.csv", index=False)
df.head()
