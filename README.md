# 🌸 Iris Flower Classification Web App

This is a **Streamlit-based web application** that allows users to classify iris flowers into one of three species: **Setosa**, **Versicolor**, or **Virginica**, based on sepal and petal length and width. The application is built using **Python** and supports multiple machine learning models.

## 📊 Models Used

This app compares predictions from several classification algorithms:

1. **Logistic Regression**
   - A linear model that estimates the probability of a class label.
2. **K-Nearest Neighbors (KNN)**
   - Classifies data points based on the majority class of their nearest neighbors.
3. **Support Vector Machine (SVM)**
   - Finds the optimal decision boundary to separate classes.
4. **Decision Tree**
   - Uses a tree-like model to make decisions based on feature values.
5. **Random Forest**
   - An ensemble method that builds multiple decision trees and averages their predictions.
6. **Naive Bayes**
   - A probabilistic classifier based on Bayes' Theorem assuming feature independence.

## 🌼 Dataset

We use the **Iris dataset (iris.csv)**, a classic dataset in machine learning that contains:

- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**
- **Species** (Setosa, Versicolor, Virginica)

## 🖥️ How to Run the App

1. Clone this repository:
   ```bash
   git clone https://github.com/Ansh07017/Optifyx-FlowerCognition.git
   cd iris-ml-streamlit
2. pip install -r requirements.txt
3. streamlit run app.py

