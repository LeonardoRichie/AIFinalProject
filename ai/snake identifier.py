#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[3]:


IMAGE_SIZE = [224,224]


# In[4]:


train_directory = 'C:\\Users\\ES\\Downloads\\artificialinteligence\\archive\\train'
test_directory = 'C:\\Users\\ES\\Downloads\\artificialinteligence\\archive\\test'


# In[5]:


# Specify the correct path to the weights file in your Kaggle input directory
weights_path = 'C:\\Users\\ES\\Downloads\\artificialinteligence\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Load VGG model with local weights
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights=weights_path, include_top=False)

# Set layers as non-trainable
for layer in vgg.layers:
    layer.trainable = False


# In[6]:


folders = glob('C:\\Users\\ES\\Downloads\\artificialinteligence\\archive\\test\\*')
len(folders)


# In[7]:


x = Flatten()(vgg.output)
prediction = Dense(len(folders),activation='softmax')(x)


# In[8]:


model = Model(inputs=vgg.input, outputs=prediction)

model.summary()


# In[9]:


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[10]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_directory,
                                                target_size = (224,224),
                                                batch_size = 32,
                                                class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_directory,
                                           target_size = (224,224),
                                           batch_size = 32,
                                           class_mode = 'categorical')


# In[11]:


print(len(training_set))
print(len(test_set))


# In[ ]:





# In[12]:


print(len(training_set))
print(len(test_set))


# In[13]:


# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()


# In[14]:


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[15]:


# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


# In[16]:


model.save('BC.h5')


# In[ ]:





# In[ ]:





# In[ ]:




