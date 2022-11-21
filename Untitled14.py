#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[8]:


plt.matshow(x_train[0])


# In[ ]:





# In[9]:


len(x_train)


# In[10]:


len(x_test)


# In[11]:


x_train.shape


# In[12]:


x_test.shape


# In[13]:


x_train[0]


# In[14]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()


# In[28]:


model.compile(optimizer="sgd",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"]
)


# In[ ]:





# In[29]:


model.summary()


# In[30]:


history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10)


# In[31]:


test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)


# In[32]:


n=random.randint(0,9999)
plt.imshow(x_test[n])
plt.show()


# In[33]:


predicted_value=model.predict(x_test)
plt.imshow(x_test[n])
plt.show()

print(predicted_value[n])


# In[34]:


history.history.keys()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[36]:


history.history.keys()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:




