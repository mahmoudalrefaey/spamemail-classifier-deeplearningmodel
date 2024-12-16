# 🚀 Spam Email Classifier using Conv1D

This repository implements a **Spam Email Classifier** using a **Convolutional Neural Network (CNN)** with a **Conv1D** architecture. The model takes in word frequency vectors extracted from emails and predicts whether the email is **Spam** or **Not Spam**.

---

## 📄 **Project Overview**

### **Objective**
To classify emails as **Spam** or **Not Spam** using word frequency vectors as input and a deep learning model built with TensorFlow/Keras.

---

## 📊 **Dataset**

The dataset represents each email as a vector where:
- **Columns**: ~3000 words (features), where each column represents a word.
- **Rows**: Individual emails.
- **Values**: The count of word occurrences in the email.

- **Dataset Source**: [Kaggle - Email Spam Classification Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

### **Labels**:
- `1` → Spam  
- `0` → Not Spam  

---

## 🛠️ **Tools & Libraries Used**
- **TensorFlow/Keras**: Model building and training.
- **Pandas**: Data preprocessing.
- **Scikit-learn**: Data scaling (StandardScaler).
- **NumPy**: Numerical computations.
- **Matplotlib**: Data visualization (optional).
- **Streamlit**: Interactive web deployment for predictions.

---

## 🧩 **Model Architecture**

1. **Input Layer**: Accepts word frequency vectors as input.
2. **Conv1D Layer(s)**: Extracts spatial patterns and relationships in word frequencies.
3. **Flatten Layer**: Prepares features for dense layers.
4. **Dense Layer(s)**: Processes extracted features for prediction.
5. **Output Layer**: Sigmoid activation for binary classification.

---

## 🌐 **Deployment on Streamlit**

The model is now deployed on **Streamlit**, where users can easily input their email text and get a prediction of whether the email is **Spam** or **Not Spam**.  
You can interact with the live model and try predictions directly on the Streamlit app:  
**[Spam Email Classifier - Streamlit App](https://your-streamlit-app-link)**

---

## 📈 **Results**
- **Training Accuracy**: ~98.1%
- **Validation Accuracy**: ~98%  

---

## 📎 **Links**
- **Dataset**: [Kaggle - Email Spam Classification Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)
- **Notebook**: [Spam Email Classifier Notebook on Kaggle](https://www.kaggle.com/code/mahmoudalrefaey/spamemails-classifier-conv1d)
- **Streamlit App**: [Spam Email Classifier on Streamlit](https://your-streamlit-app-link)

---

## 📧 **Contact for Model Access**

To access the **model.h5** file, please contact me via email at:  
**dev.mahmoudalrefaey@gmail.com**

---

## 📚 **About**
This project explores the application of deep learning for text classification using **CNN (Conv1D)**. By transforming emails into word frequency vectors, the model learns to distinguish patterns indicative of spam emails.  
The dataset and implementation offer a solid baseline for further experimentation or deployment.
