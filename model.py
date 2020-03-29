import os
import cv2
import random
random.seed(1234)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, Dropout, Activation
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, adam, adamax
from keras.utils.vis_utils import model_to_dot
import tensorflow.keras.backend as K
from IPython.display import SVG
import pydot as pyd
import pydotplus

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

### Data Loading ###
genres = ['Animation', 'Sci', 'War', 'Western']
image_genre = []
for genre in genres:
    for img in os.listdir('data/images/' + genre):
        image_genre.append((img, genre))

ig_df = pd.DataFrame(image_genre, columns=['File', 'Genre'])

path = 'data/images/'
dat = []
for img, genre in ig_df.values:
    img_array = cv2.imread(os.path.join(path + genre + '/' + img))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    reduce_img = cv2.resize(img_array, (224, 224), 3)
    dat.append([str(img).strip('.png'), reduce_img, genre])


def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        if not K.is_keras_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce


weights = [(ig_df['Genre']==x).mean() for x in genres]
loss = weighted_categorical_crossentropy(weights)
# plt.imshow(dat[100][1])
# plt.show()

### Model Building ###
X = [x[1] for x in dat]
dummies = pd.get_dummies(ig_df['Genre'])
y = [x for x in dummies.values]


keras.utils.vis_utils.pydot = pyd
def visualize_model(model):
    return SVG(model_to_dot(model).create(prog='dot', format='svg'))

filepath = 'weights.hdf5'
es = [EarlyStopping(monitor='val_categorical_accuracy', patience=10),
      ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, mode='max', period=1, save_best_only=True),
      ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=5, verbose=1, factor=0.25, min_lr=0.000000001)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1234)

input_shape=(224, 224, 3)

model = Sequential()
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(500, input_shape=(224*224*3,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(4))
model.add(Activation('softmax'))

### Compile Model ###
sgd = SGD(lr=.01, decay=1e-6)
ad = adamax(lr=0.0001)
model.compile(optimizer=ad,
             loss=loss,
             metrics=['categorical_accuracy'])

history = model.fit(np.array(X_train), np.array(y_train), epochs=100, batch_size=250,
                    validation_data=(np.array(X_test), np.array(y_test)), callbacks=es)

with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
