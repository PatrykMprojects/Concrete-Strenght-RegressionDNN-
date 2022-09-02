#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[89]:


# Link to data --> https://cocl.us/concrete_data
concrete_data = pd.read_csv('concrete_data.csv')
concrete_data.head()


# In[90]:


concrete_data.shape


# In[91]:


concrete_data.describe()


# In[92]:


concrete_data.isnull().sum()


# In[93]:


concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column


# In[94]:


predictors.head()


# In[95]:


target.head()


# In[96]:


n_cols = predictors.shape[1] # number of predictors
n_cols


# In[103]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))

    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# In[104]:


# build the model
model = regression_model()


# In[105]:


# Define a random seed to reproduce any random process
rs = 123


# In[106]:


from sklearn.model_selection import train_test_split
# Split 70% as training dataset
# and 30% as testing dataset
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state = rs)


# In[109]:


#cycle 50 times 
lst_mse = []
for c in range(50):
    #train and test model at the same time 
    scores = model.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_test, y_test))
    # MSE value in the history from our regression model
    mse = scores.history["val_loss"][-1]
    # add and display values of MSE for each cycle
    lst_mse.append(mse)
    print('Iteration number #{}: mean_squared_error {}'.format(c+1, mse))


# In[110]:


print('The mean of the mean squared errors: {}'.format(np.mean(lst_mse)))
print('The standard deviation of the mean squared errors: {}'.format(np.std(lst_mse)))


# In[111]:


# Normalize the data --> by subtracting the mean from the individual predictors and dividing by the standard deviation

norm_pred = (predictors - predictors.mean()) / predictors.std()
norm_pred.head(15)


# In[112]:


# Split 70% as training dataset
# and 30% as testing dataset
X_train, X_test, y_train, y_test = train_test_split(norm_pred, target, test_size=0.3, random_state = rs)


# In[115]:


#cycle 50 times AND TRAIN ON 100 EPOCHs
lst_mse = []
for c in range(50):
    #train and test model at the same time 
    scores = model.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))
    # MSE value in the history from our regression model
    mse = scores.history["val_loss"][-1]
    # add and display values of MSE for each cycle
    lst_mse.append(mse)
    print('Iteration number #{}: mean_squared_error {}'.format(c+1, mse))


# In[116]:


print('The mean of the mean squared errors: {}'.format(np.mean(lst_mse)))
print('The standard deviation of the mean squared errors: {}'.format(np.std(lst_mse)))


# In[ ]:


# Error decreased even more after doubling the number of epochs and normalizing data but still remains big. 


# In[120]:


# define regression model
def regression_model_1():
    # create model
    model1 = Sequential()
    model1.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model1.add(Dense(10, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    model1.add(Dense(1))

    
    # compile model
    model1.compile(optimizer='adam', loss='mean_squared_error')
    return model1


# In[121]:


# build the model
model1 = regression_model_1()


# In[124]:


# add more layers to regression model 
#cycle 50 times AND TRAIN ON 100 EPOCHs
lst_mse = []
for c in range(50):
    #train and test model at the same time 
    scores = model1.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_test, y_test))
    # MSE value in the history from our regression model
    mse = scores.history["val_loss"][-1]
    # add and display values of MSE for each cycle
    lst_mse.append(mse)
    print('Iteration number #{}: mean_squared_error {}'.format(c+1, mse))


# In[125]:


print('The mean of the mean squared errors: {}'.format(np.mean(lst_mse)))
print('The standard deviation of the mean squared errors: {}'.format(np.std(lst_mse)))


# In[ ]:


# The error decreased in comparison to the example with normalize data and 50 epochs. 
#The best results were reported in neural network without additional layers and 100 epochs

