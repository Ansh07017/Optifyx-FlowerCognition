import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from iris_model import IrisModel

# Set page configuration
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide"
)

# Initialize the model
@st.cache_resource
def load_model():
    return IrisModel()

iris_model = load_model()

# App title and introduction
st.title("🌸 Iris Species Predictor")
st.markdown("""
This application helps predict the species of iris flowers using machine learning models.
You can input the measurements of an iris flower and select a model to make predictions.
""")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Dataset Exploration", "Model Performance", "About Iris Species"])

# Tab 1: Prediction
with tab1:
    st.header("Predict Iris Species")
    
    # Create two columns for input and results
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Input Flower Measurements")
        
        # Input flower specifications using sliders
        sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.4)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.4)
        petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 1.3)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
        
        # Model selection
        model_name = st.selectbox(
            "Select a Model",
            list(iris_model.models.keys())
        )
        
        # Predict button
        predict_button = st.button("Predict Species", type="primary")
    
    with col2:
        if predict_button:
            # Prepare input data
            input_data = {
                'sepal length (cm)': sepal_length,
                'sepal width (cm)': sepal_width,
                'petal length (cm)': petal_length,
                'petal width (cm)': petal_width
            }
            
            # Predict the species
            predicted_species, probabilities = iris_model.predict_species(model_name, input_data)
            
            # Display the result
            st.subheader("Prediction Result")
            st.success(f"The predicted species is: **{predicted_species}**")
            
            # Display feature importance
            st.subheader("Feature Importance")
            feature_importance = iris_model.get_feature_importance(model_name)
            
            # Create a bar chart for feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
            
            ax.bar(features, importance, color=colors)
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance for ' + model_name)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display prediction probabilities if available
            if probabilities is not None:
                st.subheader("Prediction Confidence")
                prob_df = pd.DataFrame({
                    'Species': iris_model.target_names,
                    'Probability': probabilities
                })
                
                # Create a bar chart for probabilities
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(prob_df['Species'], prob_df['Probability'], color=['#FF9999', '#66B2FF', '#99FF99'])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities')
                st.pyplot(fig)
                
                # Show the numerical values
                st.dataframe(prob_df.set_index('Species').sort_values('Probability', ascending=False))

# Tab 2: Dataset Exploration
with tab2:
    st.header("Iris Dataset Exploration")
    
    # Display dataset statistics
    st.subheader("Dataset Overview")
    stats = iris_model.get_dataset_statistics()
    st.write(f"Dataset Shape: {stats['shape'][0]} rows, {stats['shape'][1]} columns")
    
    # Class distribution
    st.subheader("Class Distribution")
    species_counts = pd.DataFrame.from_dict(
        stats['class_distribution'], 
        orient='index', 
        columns=['Count']
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    species_counts.plot(kind='bar', ax=ax, color=['#FF9999', '#66B2FF', '#99FF99'])
    plt.title('Number of Samples per Species')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot(fig)
    
    # Show the dataset
    st.subheader("Raw Dataset")
    st.dataframe(iris_model.df)
    
    # Data visualization options
    st.subheader("Data Visualization")
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Pairplot", "Boxplot", "Violin Plot", "Correlation Heatmap"]
    )
    
    if viz_type == "Pairplot":
        fig = plt.figure(figsize=(10, 10))
        sns.pairplot(iris_model.df, hue='species_name', palette=['#FF9999', '#66B2FF', '#99FF99'])
        st.pyplot(fig)
    
    elif viz_type == "Boxplot":
        feature = st.selectbox("Select Feature", iris_model.feature_names)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='species_name', y=feature, data=iris_model.df, palette=['#FF9999', '#66B2FF', '#99FF99'])
        plt.title(f'Boxplot of {feature} by Species')
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    elif viz_type == "Violin Plot":
        feature = st.selectbox("Select Feature for Violin Plot", iris_model.feature_names)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='species_name', y=feature, data=iris_model.df, palette=['#FF9999', '#66B2FF', '#99FF99'])
        plt.title(f'Violin Plot of {feature} by Species')
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    elif viz_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = iris_model.X.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap of Features')
        st.pyplot(fig)

# Tab 3: Model Performance
with tab3:
    st.header("Model Performance Comparison")
    
    # Create a dataframe with model accuracies
    accuracies = {name: perf['accuracy'] for name, perf in iris_model.model_performances.items()}
    acc_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy'])
    
    # Plot model accuracies
    fig, ax = plt.subplots(figsize=(10, 6))
    acc_df.sort_values('Accuracy', ascending=False).plot(kind='bar', ax=ax, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Model selection for detailed metrics
    st.subheader("Detailed Model Metrics")
    selected_model = st.selectbox(
        "Select a Model for Detailed Performance Metrics",
        list(iris_model.models.keys())
    )
    
    # Get the selected model's performance
    model_perf = iris_model.get_model_performance(selected_model)
    
    # Display classification report
    st.write("Classification Report:")
    
    # Create a more visually appealing classification report
    report_data = model_perf['report']
    # Remove unnecessary keys
    if 'accuracy' in report_data:
        del report_data['accuracy']
    if 'macro avg' in report_data:
        del report_data['macro avg']
    if 'weighted avg' in report_data:
        del report_data['weighted avg']
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report_data).T
    report_df = report_df.round(2)
    st.dataframe(report_df)
    
    # Display confusion matrix
    st.write("Confusion Matrix:")
    cm = model_perf['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris_model.target_names, yticklabels=iris_model.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {selected_model}')
    st.pyplot(fig)

# Tab 4: About Iris Species
with tab4:
    st.header("About Iris Species")
    
    st.markdown("""
    ### The Iris Flower Dataset
    
    The Iris flower dataset is one of the most famous datasets in pattern recognition and machine learning. 
    It contains measurements for 150 iris flowers from three different species:
    
    - **Iris Setosa**: Known for its distinctive appearance with narrow, linear petals and sepals.
    - **Iris Versicolor**: Has more oval-shaped petals and is generally larger than Setosa.
    - **Iris Virginica**: Typically the largest of the three species with broad petals.
    
    ### Features in the Dataset
    
    The dataset includes four features measured from each flower:
    
    1. **Sepal Length**: Length of the sepal (in cm)
    2. **Sepal Width**: Width of the sepal (in cm)
    3. **Petal Length**: Length of the petal (in cm)
    4. **Petal Width**: Width of the petal (in cm)
    
    ### Importance of the Dataset
    
    The Iris dataset is widely used in machine learning as a beginner's dataset for the following reasons:
    
    - It's relatively small and easy to understand
    - It's well-structured and has no missing values
    - It involves a straightforward classification task
    - The classes have distinct patterns, making it good for teaching classification algorithms
    
    ### Interesting Facts
    
    - The dataset was introduced by the British statistician and biologist Ronald Fisher in 1936.
    - It was used to demonstrate the technique of discriminant analysis.
    - Iris Setosa is linearly separable from the other two species, while Versicolor and Virginica have some overlap.
    """)
    
    # Display visual representation of iris species characteristics
    st.subheader("Visual Comparison of Iris Species")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group data by species
    species_data = {}
    for species in iris_model.target_names:
        species_data[species] = iris_model.df[iris_model.df['species_name'] == species]
    
    # Plot average values for each feature by species
    width = 0.2
    x = np.arange(len(iris_model.feature_names))
    
    for i, (species, data) in enumerate(species_data.items()):
        means = data[iris_model.feature_names].mean().values
        ax.bar(x + i*width, means, width, label=species)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(iris_model.feature_names, rotation=45, ha='right')
    ax.set_ylabel('Average Value (cm)')
    ax.set_title('Average Measurements by Species')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.sidebar.title("About")
st.sidebar.info("""
This app demonstrates machine learning classification on the Iris dataset. 
It uses various machine learning models to predict the species of iris flowers based on their measurements.
""")

st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Go to the **Prediction** tab
2. Adjust the sliders to input flower measurements
3. Select a machine learning model
4. Click "Predict Species" to see the result
5. Explore other tabs to learn more about the dataset and models
""")

st.sidebar.title("Model Descriptions")
st.sidebar.markdown("""
- **Logistic Regression**: A linear model for classification
- **K-Nearest Neighbors**: Classification based on closest training examples
- **Support Vector Machine**: Finds the hyperplane that best separates classes
- **Decision Tree**: Uses a tree-like model of decisions
- **Random Forest**: Ensemble of decision trees
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
""")
