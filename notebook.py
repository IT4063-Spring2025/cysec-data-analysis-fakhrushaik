#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.display.max_rows = 999
warnings.filterwarnings("ignore")


# # Step 2: Loading the dataset
# 
# TODO: Load the given dataset.

# In[3]:


link = "./Data/CySecData.csv"
df = pd.read_csv(link, sep=",", encoding="latin-1")


# # Step 3: Display the first few rows of the dataset
# TODO: Import the necessary libraries for data analysis and visualization.

# In[4]:


print("First 5 rows of the dataset:")
print(df.head())


# # Step 4: Initial info on the dataset.
# 
# TODO: Provide a summary of the dataset.

# In[5]:


print("\nDataset Info:")
df.info()
print("\nStatistical Summary:")
print(df.describe())


# # Step 5: Creating dummy variables
# TODO: Create dummy variables for the categorical columns except for the label column "class".

# In[6]:


cat_cols = df.select_dtypes(include=['object']).columns.tolist()

if 'class' in cat_cols:
    cat_cols.remove('class')
    
# Create dummy variables only for the non-target categorical columns.
dfDummies = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# # Step 6: Dropping the target column
# TODO: Drop the target column 'class' from the dataset.

# In[7]:


y = dfDummies['class']
dfDummies = dfDummies.drop('class', axis=1)


# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[8]:


from sklearn.preprocessing import StandardScaler


# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[9]:


scaler = StandardScaler()
dfNormalized = scaler.fit_transform(dfDummies)


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[10]:


X = dfNormalized
# The target (y) was already separated earlier.
print("\nShapes -- Features: {}, Target: {}".format(X.shape, y.shape))


# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[11]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# # Step 11: Defining the models (Logistic Regression, Support Vector Machine, Random Forest)
# TODO: Define the models to be evaluated.

# In[12]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))




# # Step 12: Evaluating the models
# TODO: Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.
# Hint: Use Kfold cross validation and a loop

# In[13]:


print("\nModel Evaluation (10-Fold Cross Validation):")
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
for name, model in models:
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {cv_results.mean():.4f}, Standard Deviation = {cv_results.std():.4f}")


# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[ ]:


# The following command converts this Jupyter notebook to a Python script.
get_ipython().system('jupyter nbconvert --to python notebook.ipynb')

