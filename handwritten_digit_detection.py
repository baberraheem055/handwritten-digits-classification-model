

import streamlit as st

# Title
st.title("welcome to my first app")



import tensorflow as tf
import tensorflow as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer,Dense,Flatten

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()     #x_train = 6000
                                                                              # test set = 1000

x_train.shape

##to ploat a single digit
import matplotlib.pyplot as plt
plt.imshow(x_train[1])

x_train[0]

##let standarize the data to make convergence fast
X_train = x_train/255             #to centralized data between 0 and 1
X_test = x_test/255

#now lets create AAN model
model = Sequential()
model.add(Flatten(input_shape = (28,28)))    #Flatten() convert two dimention matrix into 1 dimention i.e columns into rows
model.add(Dense(128,activation = 'relu'))    #hidden layer 1         #Flatten funtion will pass total number of input to "Dense". here total nodes = 128
model.add(Dense(10,activation = "softmax"))  #hidden layer 2    #output will be 10 because there are 10 classes #here we use "softmax" activation funtion because of multiple classes outcome

model.summary()

#lets compile and train the model

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'Adam')
model.fit(X_train,y_train,epochs= 10 , validation_split = 0.2)

y_prob = model.predict(X_test)   #here model will predict the probability of each input b/w 0 , 9

#now to find out the maximum probability of digit inside each array
y_predict = y_prob.argmax(axis =  1)
y_predict
#on the output the first picture in the input has predicted as 5

#let's check the accuracy of our model
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)

plt.imshow(X_test[50])

#now to check the prediction of our model

model.predict(X_test[50].reshape(1,28,28)).argmax(axis = 1)

