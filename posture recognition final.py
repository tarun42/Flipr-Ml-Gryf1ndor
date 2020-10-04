#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.models import Sequential

from imblearn.over_sampling import SMOTE
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv('original_data.csv', header = None)


# In[4]:


data.head()


# In[5]:


y = data[[1]]
data.drop([1], axis=1,inplace = True )


# In[6]:


print(len(y))
print(len(y[y[1]==5]))
y.replace({ 1:0 , 2:0 , 3:0 , 4:0 }, inplace=True)


# In[7]:


y.replace({ 5:1  }, inplace=True)


# In[8]:


y[1].value_counts()


# In[9]:


data.drop([0],axis = 1 , inplace = True)


# In[10]:


data.columns = range(1,76)
data.head()


# In[11]:


#dropping columns which cant be detected by posenet
#Head , left shoulder , left elbow , left wrist , right shoulder , right elbow , right wrist , left hip , left knee , left ankle , right hip , right knee ,right ankle
data.drop([1,2,3,7,8,9,10,11,12,13,14,15,16,17,18,28,29,30,31,32,33,34,35,36,46,47,48,49,50,51,61,62,63,74,75,73], axis=1,inplace = True )
data.columns = range(1,40)
data.head()


# In[12]:


#data = org_data
# rescale the joints
#left hip = 22,23,24 : right hip = 31,32,33
def rescale(row): 
  #print(row)
 
    x_min = x_max =   row[1]
    z_min = z_max =  row[3]
    y_min = y_max =  row[2]

    for i in range(4,38,3):
      
      
      if(row[i]> x_max):
        x_max = row[i]
      if(row[i+1]>y_max):
        y_max = row[i+1]
      if(row[i+2]>z_max):
        z_max = row[i+2]
      if(row[i] < x_min):
        x_min = row[i]
      if(row[i+1]<y_min):
        y_min = row[i+1]
      if(row[i+2]< z_min):
        z_min = row[i+2]
    if(x_min == x_max):
      x_max+=1
    if(y_min == y_max):
      y_max+=1
    if(z_min == z_max):
      z_max+=1
    
    for i in range(1,38,3):
      row[i] = (row[i] - x_min)/(x_max - x_min)
      row[i+1] = (row[i+1] - y_min)/(y_max - y_min)
      row[i+2] = (row[i+2] - z_min)/(z_max - z_min) 
    
    return row


# In[13]:


data = data.apply(rescale , axis = 1)
data.head(5)


# In[14]:


# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(data, np.asarray(y).ravel())


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test , test_size=0.5,random_state = 42)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_val = np.asarray(x_val)
y_val = np.asarray(y_val)


# In[17]:


print(x_train.shape)
print(x_val.shape)
print(x_test.shape)


# In[19]:


e = 2 #no of epochs

model = Sequential()

model.add(Dense(523, input_shape=(39,)))

model.add(Dense(2345,activation='relu'))

model.add(Dense(4579,activation='relu'))

model.add(Dense(2567,activation='relu'))
model.add(Dense(704,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer = "adam",loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train,y_train,epochs = e ,validation_data=(x_val , y_val))


# In[27]:


epochs = e 
epoch_range = range(1, epochs+1)
plt.plot(epoch_range, history.history['accuracy'])
plt.plot(epoch_range, history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# In[29]:


_, accuracy = model.evaluate(x_test, y_test, batch_size=50, verbose=0)
print(round(accuracy*100),'%')


# In[65]:


yhat = model.predict(x_test)
for i in range(len(yhat)):
  if(yhat[i]<0.5):
    yhat[i]=0
  else:
    yhat[i]=1
print(yhat)


# In[67]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color =  "black" if(cm[i, j]<0.5) else "white" )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[68]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['fall(1)','Non - Fall(0)'],normalize= True,  title='Confusion matrix Deep Learning Model')


# In[39]:


# save model and architecture to single file
model.save("final_model.h5")
print("Saved model to disk")


# In[70]:


# load model
mymodel = tf.keras.models.load_model('final_model.h5')
# summarize model.
mymodel.summary()


# # Real Testing Data

# In[71]:


#Testing on real videos
def testing(model):
    for i in range(7,13):
        
        test = pd.read_csv('Data_'+str(i)+'.csv', header = None)
        test.drop([0],axis = 0 , inplace = True)
        test.columns = range(1,40)
        test = test.astype(np.float64)
        test.apply(rescale , axis = 1)
        y_pred = model.predict(test)
        for j in range(len(y_pred)):
          if(y_pred[j]<0.5):
            y_pred[j]=0
          else:
            y_pred[j]=1

        print(y_pred)
        print("####dataset",i," Done")       


# In[72]:


testing(mymodel)
