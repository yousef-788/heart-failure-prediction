#!/usr/bin/env python
# coding: utf-8

# ### Import Liabraries

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import warnings
warnings.filterwarnings('ignore')


# ### Import data

# In[92]:


data=pd.read_csv(r"C:\Users\m\Desktop\ML code\Final project\heart_failure_clinical_records_dataset.csv")


# ### Data Preprocessing 

# In[93]:


data.head()


# In[94]:


data.info()


# In[95]:


data.isnull().sum()


# In[96]:


data.shape


# In[97]:


corr=data.corr()
sns.heatmap(corr)


# In[98]:


data.describe()


# In[99]:


data.columns


# In[100]:


data.duplicated().sum()


# ### Removing Outliers

# In[101]:


numeric_cols = data.select_dtypes(include='number').columns
numeric_cols = numeric_cols.drop('DEATH_EVENT')
plt.figure(figsize=(16, 10))
for i, col in enumerate(data[numeric_cols].columns, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(data=data, y=col, color='lightblue')
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
plt.show()


# In[102]:


# drop outliers using Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
# shape before dropping outliers
print("Shape before dropping outliers:", data.shape)
data = data[(z_scores < 3).all(axis=1)]
# shape after dropping outliers
print("Shape after dropping outliers:", data.shape)


# ### Splitting Data

# In[103]:


x=data.drop('DEATH_EVENT',axis=1)
y=data['DEATH_EVENT']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[104]:


print(np.bincount(y))


# ### Applying PCA

# In[105]:


from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

scores = []
n_components_list = range(1, 12) 

for n in n_components_list:
    pca = PCA(n_components=n)
    pca.fit(x_train)
    variance = np.sum(pca.explained_variance_ratio_)
    scores.append(variance)

# رسم التباين المفسَّر
plt.plot(n_components_list, scores, marker='o', color='green')
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs PCA Components")
plt.grid(True)
plt.show()


# In[106]:


from sklearn.decomposition import PCA
pca=PCA(n_components=6)
x_train_pca=pca.fit_transform(x_train)
x_test_pca=pca.transform(x_test)

print(x_train_pca.shape)
print(x_test_pca.shape)


# ### Standard Normalization

# In[107]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# # MODEL

# # Logistic regression

# In[108]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

model_LR=LR=LogisticRegression(max_iter=500,solver='liblinear')
LR.fit(x_train,y_train)

y_pred = model_LR.predict(x_test)
cr=classification_report(y_test,y_pred)
print(cr)
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# # SVM

# In[109]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the base model
model = SVC()

# Define the parameter grid to search
params = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [1, 0.1, 10]
}

grid = GridSearchCV(model, params)
grid.fit(x_train, y_train)
y_pred = grid.best_estimator_.predict(x_test)
print("Best Parameters:", grid.best_params_)

cr=classification_report(y_test,y_pred)
print(cr)
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# # naive bayes

# In[110]:


from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(x_train, y_train)
y_predict = model_nb.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
cr=classification_report(y_test,y_predict)
print(cr)
sns.heatmap(confusion_matrix(y_test,y_predict), annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
acc=accuracy_score(y_test,y_predict)
print(acc)


# # KNN

# In[111]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

y_predict=knn.predict(x_test)

cr=classification_report(y_test,y_predict)
print(cr)
sns.heatmap(confusion_matrix(y_test,y_predict), annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
acc=accuracy_score(y_test,y_predict)
print(acc)


# # MLP

# ### Without PCA

# In[112]:


from tensorflow.keras import layers
import keras
model_MLP=keras.Sequential([
    layers.Dense(64, activation='relu',input_shape=(12,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

model_MLP.compile(
    loss='binary_crossentropy', 
    optimizer='adam',
    metrics=['accuracy'])
    

model_MLP.fit(x_train,y_train,epochs=10, batch_size=32,callbacks=[early_stop],validation_data=(x_test,y_test))


# ### With PCA

# In[113]:


from tensorflow.keras import layers
import keras
model=keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(6,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

model.compile(
    loss='binary_crossentropy', 
    optimizer='adam',
    metrics=['accuracy'])

model.fit(x_train_pca,y_train,epochs=50,batch_size=32,callbacks=[early_stop],validation_data=(x_test_pca,y_test))


# # AdaBoost

# # DecisionTree

# ### Without PCA

# In[114]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

stump = DecisionTreeClassifier(max_depth=1,
                               class_weight='balanced')

Boost = AdaBoostClassifier(n_estimators=50,
                           estimator=stump,
                           learning_rate=1)

model_DT = Boost.fit(x_train, y_train)

y_pred = model_DT.predict(x_test)
print(classification_report(y_test, y_pred))


# In[115]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)

print(classification_report(y_test, y_pred))


# In[116]:


import lightgbm as lgb
from sklearn.metrics import classification_report

lgb_model = lgb.LGBMClassifier(num_leaves=31, n_estimators=100, class_weight='balanced', random_state=42)

lgb_model.fit(x_train, y_train)

y_pred = lgb_model.predict(x_test)
print(classification_report(y_test, y_pred))


# ### With PCA

# In[117]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics

Boost = AdaBoostClassifier(n_estimators=50, learning_rate=0.01)
model = Boost.fit(x_train_pca, y_train)

y_pred = model.predict(x_test_pca)
print(classification_report(y_test, y_pred))


# In[118]:


import gradio as gr

def binary(val):
    return 1 if val in ["Yes", "Male"] else 0

def predict(model_choice, age, anaemia, cpk, diabetes, ef, hbp, platelets,
            serum_creatinine, ss, sex, smoking, time):
    
    anaemia = binary(anaemia)
    diabetes = binary(diabetes)
    hbp = binary(hbp)
    sex = binary(sex)
    smoking = binary(smoking)

    x = np.array([[age, anaemia, cpk, diabetes, ef, hbp, platelets,
                   serum_creatinine, ss, sex, smoking, time]])
    
    x_scaled = scaler.transform(x)

    if model_choice == "Logistic Regression":
        model = model_LR
        prob = model.predict_proba(x_scaled)[0][1]
    elif model_choice == "Decision Tree":
        model = model_DT
        prob = model.predict_proba(x_scaled)[0][1]
    else: 
        model = model_nb
        prob = model.predict_proba(x_scaled)[0][1]
    
    pred = int(prob > 0.5)
    return f"""Prediction: {'DEATH' if pred == 1 else 'SURVIVE'}
Probability of risk: {prob*100:.2f}%
Model used: {model_choice}"""

inputs = [
    gr.Radio(["Logistic Regression", "Decision Tree", "naive bayes"], label="Select Model", value="naive bayes"), 
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
    description="Enter patient data and choose model"
)

iface.launch(share=True)

