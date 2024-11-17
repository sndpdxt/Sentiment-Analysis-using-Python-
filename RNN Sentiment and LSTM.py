#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sentiment Analysis
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Load the IMDB dataset
max_features = 20000
max_len = 100
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure equal length
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# Build the LSTM model
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
score, acc = model.evaluate(X_test, y_test, batch_size=32)
print(f'Test score: {score}, Test accuracy: {acc}')


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


from tensorflow.keras.datasets import mnist


# In[9]:


(x_train,y_train), (x_test,y_test)= mnist.load_data()


# In[10]:


plt.imshow(x_train[0])


# In[13]:


y_train


# In[16]:


only_zeros=x_train[y_train==0]
plt.imshow(only_zeros[900])


# In[27]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential
np.random.seed(42)
tf.random.set_seed(42)
coding_size=100
generator=Sequential()
generator.add(Dense(100,activation='relu',input_shape=[coding_size]))
generator.add(Dense(150, activation='relu'))  # Replace 'relu' with your desired activation function
generator.add(Dense(784,activation='sigmoid'))
generator.add(Reshape((28, 28)))


# In[21]:


discriminator=Sequential()
discriminator.add(Flatten(input_shape=[28,28]))
discriminator.add(Dense(150,activation='relu'))
discriminator.add(Dense(100,activation='relu'))
discriminator.add(Dense(1,activation="sigmoid"))

discriminator.compile(loss="binary_crossentropy",optimizer="adam")


# In[29]:


#We have created a gan. Now complete the code to generate a '0' from the generator
GAN=Sequential([generator,discriminator])
discriminator.trainable=False
GAN.compile(loss="binary_crossentropy",optimizer="adam")
GAN.layers
GAN.summary()
GAN.layers[0].summary()
GAN.layers[1].summary()


# In[ ]:




