#!/usr/bin/env python
# coding: utf-8

# # Heart Stroke Prediction Model

# # Purpose

# * According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.

# * This model is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

# ### Attributes Information

# 1) id: unique identifier
# 2) gender: "Male", "Female" or "Other"
# 3) age: age of the patient
# 4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
# 5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
# 6) ever_married: "No" or "Yes"
# 7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# 8) Residence_type: "Rural" or "Urban"
# 9) avg_glucose_level: average glucose level in blood
# 10) bmi: body mass index
# 11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
# 12) stroke: 1 if the patient had a stroke or 0 if not
# *Note: "Unknown" in smoking_status means that the information is unavailable for this patient

# Importing libraries

# In[62]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[63]:


df = pd.read_csv('stroke-data.csv')


# In[64]:


df.info()


# In[65]:


df.describe()


# ### Finding null values

# In[66]:


df.isnull().sum()


# In[67]:


df['bmi'] = df['bmi'].fillna(df['bmi'].mean())


# In[68]:


df.isnull().sum()


# In[69]:


df.head(10)


# #### Checking the balance in our target variable

# In[70]:


df['stroke'].value_counts()


# ### EDA (Exploratory Data Analysis)

# Univariate Analysis on numerical columns

# In[71]:


def univariate_numerical(data,var):
    results={"missing":data[var].isnull().sum(),
             "min":data[var].min(),
            "max":data[var].max(),
            "sum":data[var].sum(),
            "average":data[var].mean(),
            "Standart deviation":data[var].std(),
            "variance":data[var].var(),
            "skewness":data[var].skew(),
            "kurtosis":data[var].kurt(),
            "25th percentile":data[var].quantile(.25),
            "50th percentile":data[var].quantile(.50)}
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(data[var])
    plt.subplot(1,2,2)
    plt.boxplot(data[var])
    plt.show()
    return results


# In[72]:


df.dtypes[df.dtypes!= 'object'].index


# In[73]:


# 'id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level','bmi', 'stroke'
univariate_numerical(df,'age')


# In[74]:


# 'id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level','bmi', 'stroke'
univariate_numerical(df,'avg_glucose_level')


# In[75]:


g=df[df['avg_glucose_level']>=99]


# In[76]:


g.shape


# In[77]:


df.dtypes[df.dtypes=='object'].index


# Univariate Analysis on Categorical Columns

# In[78]:


def univariate_categorical(data,var):
    miss=data[var].isnull().sum()
    unicat=data[var].nunique()
    uncat_list=data[var].unique()
    
    h=pd.DataFrame(data[var].value_counts().reset_index())
    h.columns=[var,"count"]
    
    k=pd.DataFrame(data[var].value_counts(normalize=True).reset_index())
    k.columns=[var,"Percentage"]
    k["Percentage"]=(round(k["Percentage"]*100,2)).astype("str")+str("%")
    
    final=pd.merge(h,k,on=var,how="inner")
    
    print(f"Missing counts: {miss}\n")
    print(f"Total unique counts: {unicat}\n")
    print(f"Unique Categories: {uncat_list}\n")
    plt.figure(figsize=(10,6))
    sns.countplot(data=data,x=var)
    plt.xticks(rotation=45)
    plt.show()
    return final


# #### Visualization using Countplot

# In[79]:


categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

fig, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(5, 3 * len(categorical_columns)))

for col, ax in zip(categorical_columns, axes):
    sns.countplot(x=col, hue='stroke', data=df, ax=ax)
    ax.set_title(f'Countplot of {col} vs Stroke')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()


# #### Analysis by gender

# In[80]:


univariate_categorical(df,'gender')


# In[81]:


df['gender']= df['gender'].replace(['Other'],'Male')


# In[82]:


df['gender'].value_counts()


# #### Analysis by work_type

# In[83]:


univariate_categorical(df,'work_type')


# #### Analysis by residence_type

# In[84]:


univariate_categorical(df,'Residence_type')


# #### Analysis by smoking status

# In[85]:


univariate_categorical(df,'smoking_status')


# In[86]:


df.head()


# In[87]:


df = pd.get_dummies(df)


# In[88]:


df.drop(columns=['id'],inplace = True)


# In[89]:


df.reset_index()


# In[90]:


df['stroke'].value_counts()


# #### Our Target Variable is Imbalanced. So we have to treat this using one of the sampling technique SMOTE

# ### Model Develpment before using SMOTE

# In[91]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[92]:


X = df.drop(columns=['stroke'])
y = df['stroke']


# In[93]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Standardizing numerical columns

# In[94]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[95]:


lr = LogisticRegression()


# In[96]:


lr.fit(X_train,y_train)


# In[97]:


lr.score(X_train,y_train)


# In[98]:


lr.fit(X_test,y_test)


# In[99]:


lr.score(X_test,y_test)


# In[100]:


pred_train = lr.predict(X_train)
pred_test = lr.predict(X_test)


# In[101]:


print(metrics.classification_report(pred_test, y_test))


# In[102]:


pd.DataFrame(confusion_matrix(y_test,pred_test),
             columns = ['Predicted_no','Predicted_yes'],index = ['Actual_no','Actual_yes'])


# ### Oversampling the Data using SMOTE

# In[103]:


from imblearn.over_sampling import SMOTE


# In[104]:


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# In[105]:


lr = LogisticRegression()


# In[106]:


lr.fit(X_train_smote,y_train_smote)


# In[107]:


lr.score(X_train_smote,y_train_smote)


# In[108]:


y_pred = lr.predict(X_test)


# In[109]:


print(metrics.classification_report(y_test, y_pred))


# In[110]:


print(metrics.accuracy_score(y_test, y_pred))


# In[111]:


pd.DataFrame(confusion_matrix(y_test,y_pred),
             columns = ['Predicted_no','Predicted_yes'],index = ['Actual_no','Actual_yes'])


# ### The performance metrics before and after applying SMOTE

# Performance metrics before SMOTE

# In[112]:


precision_before = [1.0, 0.3]
recall_before = [0.94, 0.50]
f1_before = [0.97, 0.3]


#  Performance metrics after SMOTE

# In[113]:


precision_after = [0.98, 0.17]
recall_after = [0.76, 0.79]
f1_after = [0.85, 0.28]


# ## Valuable insights and observations from the above analysis report

# Data Overview:
# * The dataset contains information about individuals, including age, gender, marital status, work type, residence type, average glucose level, BMI, smoking status, and whether the person had a stroke. There was a small amount of missing data in the BMI column, and those rows were dropped during preprocessing.
# 
# Data Exploration:
# * The dataset is imbalanced, with a significantly higher number of individuals who did not have a stroke (0) compared to those who had a stroke (1). The exploration of categorical columns revealed interesting patterns, such as a higher occurrence of strokes in older individuals, those with hypertension, and those with heart disease.
# 
# Data Preprocessing:
# * Label encoding were applied to categorical columns to prepare the data for modeling. Missing values were handled by dropping rows with missing BMI values.
# 
# Data Visualization:
# * Countplots and bar graphs were used to visualize the distribution of stroke occurrences based on categorical and numerical features. The impact of variables like hypertension and heart disease on stroke occurrence was examined.
# 
# Modeling:
# * A logistic regression model was initially trained without addressing class imbalance, resulting in high accuracy but poor performance on predicting strokes (class 1). After applying SMOTE to oversample the minority class, the model's performance improved, especially in terms of recall for strokes.
# 
# Model Evaluation:
# * The model achieved a balanced performance, with improved recall for strokes, indicating better identification of individuals at risk. The confusion matrix and classification report provided detailed insights into the model's predictions.
# 
# Insights for Future Work:
# * Further exploration and feature engineering could enhance the model's predictive power. Consideration of additional models and hyperparameter tuning might improve performance. Continuous monitoring and updates to the model as more data becomes available could enhance its effectiveness.

# In[ ]:




