#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[126]:


X = pd.read_csv("../assets/data/data_new.csv", header=None, index_col=None)
if X.columns.to_list().count('target') > 0:
    X = X.drop('target', axis=1)
Y = pd.read_csv("../assets/data/targets.csv", index_col=0)


# In[3]:


Y = pd.DataFrame(1*(Y.target > 0), columns=['target'])


# In[4]:


Y.target.value_counts()


# Since here we have a good amount of data, lets use deep learning

# In[77]:


import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error


# In[22]:


encoding_dim = 3

input_shape = tf.keras.Input(shape=(100,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_shape)
decoded = tf.keras.layers.Dense(100, activation='sigmoid')(encoded)

autoencoder = tf.keras.Model(input_shape, decoded)
encoder = tf.keras.Model(input_shape, encoded)


# In[23]:


scaler = MinMaxScaler()
X_ = scaler.fit_transform(X)


# In[24]:


X_.shape, Y.shape


# In[25]:


X_norm = X_[( Y.target < 1 )]
X_anom = X_[( Y.target > 0 )]


# In[26]:


X_train, X_val = train_test_split(X_norm, test_size=0.1)


# In[27]:


X_train.shape


# In[28]:


autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
autoencoder.fit(X_train, X_train, epochs=500,
                batch_size=20,
                shuffle=True,
                validation_data=(X_val, X_val),
               callbacks=tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True))


# In[33]:


enc_anom = encoder.predict(X_anom)


# In[34]:


enc_norm = encoder.predict(X_norm)


# In[35]:


enc_norm.shape, enc_anom.shape


# In[36]:


X_mix = np.row_stack((enc_norm, enc_anom))
Y_mix = np.row_stack((np.array([0]*enc_norm.__len__()).reshape(-1,1), np.array([1]*enc_anom.__len__()).reshape(-1,1)))


# In[37]:


Y_mix.shape, X_mix.shape


# <b> Lets check the reconstruction errors </b>

# In[78]:


mean_absolute_error(X_anom, autoencoder.predict(X_anom))


# In[79]:


mean_absolute_error(X_norm, autoencoder.predict(X_norm))




def anomaly(sample, threshold=0.05):
    if mean_absolute_error(sample.transpose(), autoencoder.predict(sample.reshape(1,-1)).transpose()) > threshold:
        return 1
    else: 
        return 0


# In[118]:


rand_idx = np.random.randint(0, X_anom.__len__(), 1)
anomaly(X_anom[rand_idx])




pred = []
for x in X_anom:
    pred.append(anomaly(x))
print(f"Accuracy for anomalies: {sum(pred)/pred.__len__()}")


