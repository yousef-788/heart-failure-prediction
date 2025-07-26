# Heart Failure Prediction

This repository contains a complete machine learning project for predicting the likelihood of heart failure based on patient clinical data.

## 🖼️ Project Screenshot

> *(Add screenshot here)*  
> يمكنك رفع صورة لواجهة Gradio أو نموذج النتائج هنا لتوضيح شكل التطبيق.

## 📊 Dataset

We used the **Heart Failure Clinical Records Dataset** from the UCI Machine Learning Repository. It includes 13 clinical features such as age, serum creatinine, ejection fraction, and more, with a binary target indicating whether the patient experienced a heart failure event.

## 🛠️ Features

- Data loading and preprocessing
- Training and testing multiple classification models
- Evaluation using accuracy and confusion matrix
- User-friendly web interface with Gradio

## 🧠 Machine Learning Models Used

The following ML algorithms were implemented and compared:

- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- AdaBoost  
- LightGBM  
- Naive Bayes  
- Multi-Layer Perceptron (MLP)

## 🚀 How to Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/yousef-788/heart-failure-prediction.git
   cd heart-failure-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Gradio app:
   ```
   python app.py
   ```

This will open a local Gradio interface in your browser for real-time prediction.

## 📁 Project Structure

- `app.py` – The main Python script running the ML model and Gradio interface  
- `app.ipynb` – Jupyter notebook used for development and exploration  
- `heart_failure_clinical_records_dataset.csv` – The dataset  
- `requirements.txt` – List of required Python libraries

## 🔍 Input Features

The app takes the following clinical inputs:

- Age  
- Anaemia (0 or 1)  
- Creatinine Phosphokinase  
- Diabetes (0 or 1)  
- Ejection Fraction  
- High Blood Pressure (0 or 1)  
- Platelets  
- Serum Creatinine  
- Serum Sodium  
- Sex (0 = female, 1 = male)  
- Smoking (0 or 1)  
- Time (follow-up period in days)

## ✅ Output

The output is a binary prediction:
- `1`: High risk of heart failure  
- `0`: Low risk of heart failure

## 🧰 Technologies Used

- Python  
- Pandas, NumPy, Scikit-learn  
- Gradio (for UI)  
- Jupyter Notebook

## 👨‍💻 Author

[Yousef Hamdy](https://www.linkedin.com/in/yousef-hamdy-ee/)

---

Feel free to fork this repository, use it, or improve it!
