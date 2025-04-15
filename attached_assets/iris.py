import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df=load_iris(as_frame=True)
df=pd.DataFrame(data=df.data,columns=df.feature_names)
df['species']=load_iris(as_frame=True).target
df['species']=df['species'].map({0:'setosa',1:'versicolor',2:'virginica'})
df.head()
df.info()
df.describe()
df.isnull().sum()
df['species'].value_counts()
X = df.drop('species', axis=1)
y = df['species']

# Feature scaling/normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model Selection and Training

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}
# Train models and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Function to predict species based on input specifications
def predict_species(model, input_data):
    # Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    input_data_scaled = scaler.transform(input_df)  # Scale the input data
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Example usage
# Select a model (e.g., Random Forest)
selected_model = models["Random Forest"]

# Input specifications for prediction (matching dataset columns)
# Define the range of each specification
spec_ranges = {
    'sepal length (cm)': (4.3, 7.9),
    'sepal width (cm)': (2.0, 4.4),
    'petal length (cm)': (1.0, 6.9),
    'petal width (cm)': (0.1, 2.5)
}

# Take input from the user
input_data = {}
for spec, (min_val, max_val) in spec_ranges.items():
    while True:
        try:
            value = float(input(f"Enter {spec} (range {min_val} - {max_val}): "))
            if min_val <= value <= max_val:
                input_data[spec] = value
                break
            else:
                print(f"Please enter a value within the range {min_val} - {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Predict the species
predicted_species = predict_species(selected_model, input_data)
print(f"The predicted species is: {predicted_species}")
