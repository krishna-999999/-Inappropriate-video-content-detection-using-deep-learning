import cv2
import os
import numpy as np
import pandas as pd
from keras.layers import Bidirectional, LSTM
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from keras_efficientnets import EfficientNetB7
from keras.callbacks import ModelCheckpoint 

from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
'''
X = []
Y = []

for root, dirs, directory in os.walk('Dataset'):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32, 32))
            label = 0
            if name == 'violence':
                label = 1
            X.append(img)
            Y.append(label)
            print(name+" "+directory[j]+" "+str(label))
        
X = np.asarray(X)
Y = np.asarray(Y)
print(Y)

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

X = X.astype('float32')
X = X/255


indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and tesrt
#eb = EfficientNetB7(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights=None)
#eb.trainable = False
cnn_model = Sequential()
#cnn_model.add(eb)
cnn_model.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
if os.path.exists("model/model_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model = load_model("model/model_weights.hdf5")

cnn_model = Model(cnn_model.inputs, cnn_model.layers[-2].output)#creating cnn model
cnn_features = cnn_model.predict(X)  #extracting cnn features from test data
cnn_features = np.reshape(cnn_features, (cnn_features.shape[0], 16, 16))
print(cnn_features.shape)

X_train, X_test, y_train, y_test = train_test_split(cnn_features, Y, test_size=0.2)
bilstm_model = Sequential() #defining deep learning sequential object
#adding LSTM bidirectional layer with 32 filters to filter given input X train data to select relevant features
bilstm_model.add(Bidirectional(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
#adding dropout layer to remove irrelevant features
bilstm_model.add(Dropout(0.2))
#adding another layer
bilstm_model.add(Bidirectional(LSTM(32)))
bilstm_model.add(Dropout(0.2))
#defining output layer for prediction
bilstm_model.add(Dense(y_train.shape[1], activation='softmax'))
#compile GRU model
bilstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#start training model on train data and perform validation on test data
if os.path.exists("model/bilstm_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/bilstm_weights.hdf5', verbose = 1, save_best_only = True)
    hist = bilstm_model.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)    
else:
    bilstm_model = load_model("model/bilstm_weights.hdf5")


predict = bilstm_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, predict)
print(acc)





