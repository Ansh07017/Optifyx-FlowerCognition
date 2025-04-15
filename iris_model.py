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
from sklearn.inspection import permutation_importance

class IrisModel:
    def __init__(self):
        # Load and prepare the dataset
        self.load_data()
        self.prepare_data()
        self.train_models()
        
    def load_data(self):
        """Load the Iris dataset"""
        iris = load_iris(as_frame=True)
        self.df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.df['species'] = iris.target
        self.df['species_name'] = iris.target_names[iris.target]
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
    def prepare_data(self):
        """Prepare data for training"""
        self.X = self.df.drop(['species', 'species_name'], axis=1)
        self.y = self.df['species_name']
        
        # Feature scaling/normalization
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Train-test split (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
    
    def train_models(self):
        """Train multiple ML models"""
        # Initialize models
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Naive Bayes": GaussianNB()
        }
        
        # Train models and store performance metrics
        self.model_performances = {}
        
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            # Store performance metrics
            self.model_performances[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'report': classification_report(self.y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
    
    def predict_species(self, model_name, input_data):
        """Predict species based on input specifications"""
        # Convert input dictionary to a DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale the input data
        input_data_scaled = self.scaler.transform(input_df)
        
        # Get the selected model
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        
        # Get prediction probabilities if model supports it
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data_scaled)[0]
        
        return prediction[0], probabilities
    
    def get_feature_importance(self, model_name):
        """Get feature importance for the selected model"""
        model = self.models[model_name]
        
        # Different models have different methods to get feature importance
        if model_name in ["Random Forest", "Decision Tree"]:
            importance = model.feature_importances_
            return dict(zip(self.feature_names, importance))
        
        # For models without direct feature_importances_, use permutation importance
        else:
            result = permutation_importance(
                model, self.X_test, self.y_test, n_repeats=10, random_state=42
            )
            return dict(zip(self.feature_names, result.importances_mean))
    
    def get_model_performance(self, model_name):
        """Get performance metrics for the selected model"""
        return self.model_performances[model_name]
    
    def get_dataset_statistics(self):
        """Get basic statistics for the Iris dataset"""
        return {
            'shape': self.df.shape,
            'description': self.df.describe(),
            'class_distribution': self.df['species_name'].value_counts().to_dict()
        }
