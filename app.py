#!/usr/bin/env python
# coding: utf-8

# ### Import Liabraries

# In[386]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


# ### Import data

# In[387]:


data=pd.read_csv(r"C:\Users\m\Desktop\ML code\Final project\heart_failure_clinical_records_dataset.csv")


# ### Data Preprocessing 

# In[388]:


data.head()


# In[389]:


data.info()


# In[390]:


data.isnull().sum()


# In[391]:


data.shape


# In[ ]:


corr=data.corr()
sns.heatmap(corr)


# In[393]:


data.describe()


# In[394]:


data.columns


# In[395]:


data.duplicated().sum()


# ### Removing Outlayers

# In[396]:


# drop outliers using Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
# shape before dropping outliers
print("Shape before dropping outliers:", data.shape)
df = data[(z_scores < 3).all(axis=1)]
# shape after dropping outliers
print("Shape after dropping outliers:", df.shape)


# ### Splitting Data

# In[397]:


x=df.drop('DEATH_EVENT',axis=1)
y=df['DEATH_EVENT']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[398]:


print(np.bincount(y))


# ### Standard Normalization

# In[399]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# ### Applying PCA

# In[400]:


from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x_train)

variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(variance) + 1), variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Variance')
plt.grid()
plt.show()


# In[401]:


from sklearn.decomposition import PCA
pca=PCA(n_components=3)
x_train_pca=pca.fit_transform(x_train)
x_test_pca=pca.transform(x_test)

print(x_train_pca.shape)
print(x_test_pca.shape)


# # MODEL

# # Using Logistic regression

# In[402]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

model_LR=LR=LogisticRegression(max_iter=1000,solver='lbfgs')
LR.fit(x_train,y_train)

y_pred = model_LR.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])


# # Using MLP

# ### Without PCA

# In[414]:


from tensorflow.keras import layers
import keras
model=keras.Sequential([
    layers.Dense(64, activation='relu',input_shape=(12,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

model.compile(
    loss='binary_crossentropy', 
    optimizer='adam',
    metrics=['accuracy'])
    

model.fit(x_train,y_train,epochs=50, batch_size=32,callbacks=[early_stop],validation_data=(x_test,y_test))


# ### With PCA

# In[ ]:


# from tensorflow.keras import layers
# import keras
# model=keras.Sequential([
#     layers.Dense(128, activation='relu', input_shape=(3,)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
# from tensorflow.keras.callbacks import EarlyStopping

# early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

# model.compile(
#     loss='binary_crossentropy', 
#     optimizer='adam',
#     metrics=['accuracy'])

# model.fit(x_train_pca,y_train,epochs=50,batch_size=32,callbacks=[early_stop],validation_data=(x_test_pca,y_test))


# # Using AdaBoost

# ### Without PCA

# In[ ]:


# from sklearn.ensemble import AdaBoostClassifier
# from sklearn import datasets
# from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier

# stump = DecisionTreeClassifier(max_depth=1,
#                                class_weight='balanced')

# Boost = AdaBoostClassifier(n_estimators=50,
#                            estimator=stump,
#                            learning_rate=1)

# model = Boost.fit(x_train, y_train)

# y_pred = model.predict(x_test)
# print(classification_report(y_test, y_pred))


# ### With PCA

# In[ ]:


# from sklearn.ensemble import AdaBoostClassifier
# from sklearn import datasets
# from sklearn import metrics

# Boost = AdaBoostClassifier(n_estimators=50, learning_rate=0.01)
# model = Boost.fit(x_train_pca, y_train)

# y_pred = model.predict(x_test_pca)
# print(classification_report(y_test, y_pred))


# In[417]:


import numpy as np
import gradio as gr

# دالة تحويل نعم/لا أو ذكر/أنثى إلى 0/1
def binary(val):
    return 1 if val in ["Yes", "Male"] else 0

# دالة التوقع
def predict(age, anaemia, cpk, diabetes, ef, hbp, platelets, serum_creatinine, ss, sex, smoking, time):
    # نحول كل المدخلات البوليانية إلى أرقام قبل بناء المصفوفة
    anaemia = binary(anaemia)
    diabetes = binary(diabetes)
    hbp = binary(hbp)
    sex = binary(sex)
    smoking = binary(smoking)

    x = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets,
                   serum_creatinine, ss, sex, smoking, time]])

    x_scaled = scaler.transform(x)
    prob = model_LR.predict_proba(x_scaled)[0][1]
    pred = int(prob > 0.5)
    return f"Prediction: {'DEATH' if pred == 1 else 'SURVIVE'}\nProbability: {prob:.2f}"

# مداخل Gradio - نستخدم strings بدل أرقام
inputs = [
    gr.Slider(30, 100, value=60, label="Age"),
    gr.Radio(["No", "Yes"], label="Anaemia", value="No"),
    gr.Slider(20, 8000, value=250, label="Creatinine Phosphokinase"),
    gr.Radio(["No", "Yes"], label="Diabetes", value="No"),
    gr.Slider(10, 80, value=40, label="Ejection Fraction"),
    gr.Radio(["No", "Yes"], label="High Blood Pressure", value="No"),
    gr.Slider(25000, 850000, value=250000, label="Platelets"),
    gr.Slider(0.1, 10.0, value=1.0, label="Serum Creatinine"),
    gr.Slider(110, 150, value=135, label="Serum Sodium"),
    gr.Radio(["Female", "Male"], label="Sex", value="Male"),
    gr.Radio(["No", "Yes"], label="Smoking", value="No"),
    gr.Slider(0, 300, value=100, label="Follow-up Time (days)"),
]

iface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="Heart Failure Risk Predictor",
    description="Enter patient data"
)

iface.launch()

