
# coding: utf-8

# In[1]:


#Import required modules
import numpy as np
import pandas as pd
import scipy
from math import sqrt
import matplotlib.pyplot as plt


# In[3]:


#import estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model


# In[4]:


#import model metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[5]:


#import Cross Validation
from sklearn.cross_validation import train_test_split


# In[10]:


#Import the the raw data from CSV file
credit = pd.read_csv('D:/Python Data Science UTA/default of credit card clients.csv', header =1)
credit.head()


# In[11]:


#Examine the imported raw data from cerdit default file.
credit.info()
credit.describe()


# In[12]:


#Raw data loaded from the cedit default file is  split into features and dependent variable. Then we will create X_Train, Y_Train, X_Test and Y_Test data sets for model building
#Feature selection
features = credit.iloc[:,12:23]
print('Summary of feature sample')
features.head()


# In[14]:


#Dependent variable selection
depVar = credit['PAY_AMT6']


# In[15]:


#Establish the training set for the X-variables or Feature space (first 1000 rows: only for this example you will still follow a 70/30 split for your final models)
#Training Set (Feature Space: X Training)
X_train = (features[: 1000])
X_train.head()


# In[16]:


#Establish the training set for the Y-variable or dependent variable (the number of rows much match the X-training set)
#Dependent Variable Training Set (y Training)
y_train = depVar[: 1000]
y_train_count = len(y_train.index)
print('The number of observations in the Y training set are:',str(y_train_count))
y_train.head()


# In[17]:


#Establish the testing set for the X-Variables or Feature space
#Testing Set (X Testing)
X_test = features[-100:]
X_test_count = len(X_test.index)
print('The number of observations in the feature testing set is:',str(X_test_count))
print(X_test.head())


# In[18]:


#Establish Ground truth
#Ground Truth (y_test) 
y_test = depVar[:-100]
y_test_count = len(y_test.index)
print('The number of observations in the Y training set are:',str(y_test_count))
y_test.head()


# In[19]:


#implement Cross Validation by running the following on the X and Y training sets:
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)


# In[20]:


#use the shape function to check that the split was made as needed:
X_train.shape, X_test.shape


# In[21]:


#Established three different models with the individual variable names
#Models
modelSVR = SVR()
modelRF = RandomForestRegressor()
modelLR = LinearRegression()


# In[27]:


#Fit above three models to our train datasets.
#Support Vector Regression
modelSVR.fit(X_train,y_train)
print(cross_val_score(modelSVR, X_train, y_train)) 
modelSVR.score(X_train,y_train)


# In[25]:


#Random Forest
modelRF.fit(X_train,y_train)
print(cross_val_score(modelRF, X_train, y_train))
modelRF.score(X_train,y_train)


# In[26]:


#Linear Regression
modelLR.fit(X_train,y_train)
print(cross_val_score(modelLR, X_train, y_train)) 
modelLR.score(X_train,y_train)


# In[28]:


##From the model score random forest is the best model out of three.
#Making Predictions using RF 
predictions = modelRF.predict(X_test)


# In[32]:


#calculating RMSE
rmse = sqrt(mean_squared_error(y_test, predictions))
print('RMSE: %.3f' % rmse)


# In[33]:


#Calculate Rquared value using r2_score function, y_test and predections from previous step.
predRsquared = r2_score(y_test,predictions)
print('R Squared: %.3f' % predRsquared)


# In[36]:


#Plotting the Results, create scatterplot using matplotlib pyplot.
plt.scatter(y_test, predictions, color=['blue','green'], alpha = 0.5)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.show();

