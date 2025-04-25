#streamlit run app.py --server.port 8501 --server.address 127.0.0.1
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply simple, modern styling
st.markdown("""
<style>
    .main-title {
        color: #FF4B4B;
        font-size: 2.5rem;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .section-title {
        color: #4B4BFF;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .highlight-text {
        color: #FF4B4B;
        font-weight: bold;
    }
    
    /* Clean up button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the model
@st.cache_resource
def load_model():
    return IrisModel()

iris_model = load_model()

# App title and introduction
st.markdown("<h1 class='main-title'>🌸 Iris Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <p>Welcome to the Iris Species Predictor! This interactive application uses machine learning
    to predict the species of iris flowers based on their measurements.</p>
    <p>The app is powered by several ML models that have been trained on the famous Iris dataset.
    You can select any model and compare their performance.</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Dataset Exploration", "Model Performance", "About Iris Species"])

# Tab 1: Prediction
with tab1:
    st.markdown("<h2 class='section-title'>Predict Iris Species</h2>", unsafe_allow_html=True)
    
    # Create two columns for input and results
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-title' style='font-size: 1.2rem;'>Input Flower Measurements</h3>", unsafe_allow_html=True)
        
        # Input flower specifications using sliders with custom styling
        st.markdown("<p style='margin-bottom: 0.2rem; font-weight: bold; color: #5566cc;'>Sepal Length (cm)</p>", unsafe_allow_html=True)
        sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.4, key="sepal_length_slider", label_visibility="hidden")
                
        st.markdown("<p style='margin-bottom: 0.2rem; font-weight: bold; color: #5566cc;'>Sepal Width (cm)</p>", unsafe_allow_html=True)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.4, key="sepal_width_slider", label_visibility="hidden")
                
        st.markdown("<p style='margin-bottom: 0.2rem; font-weight: bold; color: #5566cc;'>Petal Length (cm)</p>", unsafe_allow_html=True)
        petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 1.3, key="petal_length_slider", label_visibility="hidden")
                
        st.markdown("<p style='margin-bottom: 0.2rem; font-weight: bold; color: #5566cc;'>Petal Width (cm)</p>", unsafe_allow_html=True)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, key="petal_width_slider", label_visibility="hidden")
        
        # Summary of input data
        st.markdown("<h3 class='sub-header' style='font-size: 1.2rem; margin-top: 1.5rem;'>Measurement Summary</h3>", unsafe_allow_html=True)
        input_summary = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
        }).set_index('Feature')
        st.dataframe(input_summary, use_container_width=True)
        
        # Model selection with enhanced styling
        st.markdown("<h3 class='sub-header' style='font-size: 1.2rem; margin-top: 1.5rem;'>Select Machine Learning Model</h3>", unsafe_allow_html=True)
        model_name = st.selectbox(
            "",
            list(iris_model.models.keys()),
            key="model_selector"
        )
        
        # Predict button with enhanced styling
        st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
        predict_button = st.button("📊 Predict Iris Species", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
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
            
            # Display the result with enhanced styling
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-header'>Prediction Result</h3>", unsafe_allow_html=True)
            
            # Display species with a more attractive visual
            species_colors = {
                'setosa': '#FF9999',
                'versicolor': '#66B2FF',
                'virginica': '#99FF99'
            }
            species_color = species_colors.get(predicted_species.lower(), '#FF4B4B')
            
            st.markdown(f"""
            <div class='prediction-result' style='background-color: {species_color}20; border: 2px solid {species_color}; margin-bottom: 1.5rem;'>
                <p>The predicted species is:</p>
                <h2 style='color: {species_color};'>Iris {predicted_species}</h2>
                <p>Predicted using {model_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display feature importance with improved visuals
            st.markdown("<h3 class='sub-header'>Feature Importance</h3>", unsafe_allow_html=True)
            st.markdown("<p>Which measurements were most important for this prediction?</p>", unsafe_allow_html=True)
            feature_importance = iris_model.get_feature_importance(model_name)
            
            # Create a bar chart for feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
            
            # Sort by importance
            sorted_indices = np.argsort(importance)[::-1]
            sorted_features = [features[i] for i in sorted_indices]
            sorted_importance = [importance[i] for i in sorted_indices]
            
            bars = ax.barh(sorted_features, sorted_importance, color=colors)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance for ' + model_name)
            
            # Add value labels to the bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                       ha='left', va='center', fontweight='bold')
                
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display prediction probabilities if available with improved visuals
            if probabilities is not None:
                st.markdown("<h3 class='sub-header'>Prediction Confidence</h3>", unsafe_allow_html=True)
                st.markdown("<p>How confident is the model in its prediction?</p>", unsafe_allow_html=True)
                
                prob_df = pd.DataFrame({
                    'Species': iris_model.target_names,
                    'Probability': probabilities
                })
                
                # Create a bar chart for probabilities with improved styling
                fig, ax = plt.subplots(figsize=(10, 5))
                species_list = prob_df['Species']
                probs = prob_df['Probability']
                
                # Use species-specific colors
                species_colors = ['#FF9999', '#66B2FF', '#99FF99']
                
                # Sort by probability
                sorted_indices = np.argsort(probs)[::-1]
                sorted_species = [species_list[i] for i in sorted_indices]
                sorted_probs = [probs[i] for i in sorted_indices]
                sorted_colors = [species_colors[i] for i in sorted_indices]
                
                bars = ax.barh(sorted_species, sorted_probs, color=sorted_colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Probabilities')
                
                # Add percentage labels to the bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                           ha='left', va='center', fontweight='bold')
                
                st.pyplot(fig)
                
                # Show the numerical values with improved formatting
                formatted_probs = prob_df.copy()
                formatted_probs['Confidence'] = formatted_probs['Probability'].apply(lambda x: f"{x:.2%}")
                formatted_probs = formatted_probs.sort_values('Probability', ascending=False)
                formatted_probs = formatted_probs[['Confidence']]
                st.dataframe(formatted_probs, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Dataset Exploration
with tab2:
    st.markdown("<h2 class='sub-header'>Iris Dataset Exploration</h2>", unsafe_allow_html=True)
    
    # Display dataset statistics in a box
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>Dataset Overview</h3>", unsafe_allow_html=True)
    stats = iris_model.get_dataset_statistics()
    
    # Create a more visual dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style='background-color: #FF99992a; padding: 1rem; border-radius: 10px; text-align: center; height: 100%;'>
            <h4 style='margin: 0; color: #990000;'>Total Samples</h4>
            <p style='font-size: 2rem; font-weight: bold; margin: 0.5rem 0;'>{stats['shape'][0]}</p>
            <p>iris flowers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color: #66B2FF2a; padding: 1rem; border-radius: 10px; text-align: center; height: 100%;'>
            <h4 style='margin: 0; color: #000099;'>Features</h4>
            <p style='font-size: 2rem; font-weight: bold; margin: 0.5rem 0;'>{stats['shape'][1]-2}</p>
            <p>measurements</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background-color: #99FF992a; padding: 1rem; border-radius: 10px; text-align: center; height: 100%;'>
            <h4 style='margin: 0; color: #009900;'>Classes</h4>
            <p style='font-size: 2rem; font-weight: bold; margin: 0.5rem 0;'>{len(stats['class_distribution'])}</p>
            <p>iris species</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Class distribution with enhanced visuals
    st.markdown("<h3 class='sub-header' style='margin-top: 1.5rem;'>Class Distribution</h3>", unsafe_allow_html=True)
    st.markdown("<p>Number of samples for each iris species in the dataset:</p>", unsafe_allow_html=True)
    
    species_counts = pd.DataFrame.from_dict(
        stats['class_distribution'], 
        orient='index', 
        columns=['Count']
    )
    
    # Create a bar chart with improved styling
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = species_counts.plot(kind='bar', ax=ax, color=['#FF9999', '#66B2FF', '#99FF99'])
    plt.title('Number of Samples per Species', fontsize=16, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0, fontsize=12)
    
    # Add count labels on top of bars
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            str(int(bar.get_height())),
            ha='center', fontweight='bold'
        )
    
    st.pyplot(fig)
    
    # Dataset viewer with improved styling
    st.markdown("<h3 class='sub-header' style='margin-top: 1.5rem;'>Dataset Preview</h3>", unsafe_allow_html=True)
    st.markdown("<p>Explore the first 10 rows of the Iris dataset:</p>", unsafe_allow_html=True)
    
    # Format the dataframe for better display
    display_df = iris_model.df.head(10).copy()
    # Add styling to the dataframe
    st.dataframe(display_df, use_container_width=True, height=300)
    
    # Add a download button for the dataset
    st.download_button(
        label="📥 Download Full Dataset",
        data=iris_model.df.to_csv(index=False).encode('utf-8'),
        file_name='iris_dataset.csv',
        mime='text/csv'
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Data visualization options with enhanced styling
    st.markdown("<div class='card' style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Interactive Data Visualization</h3>", unsafe_allow_html=True)
    st.markdown("<p>Explore the Iris dataset through different visualization techniques:</p>", unsafe_allow_html=True)
    
    # Visualization selector with tabs for better organization
    viz_tabs = st.tabs(["Pairplot", "Boxplot", "Violin Plot", "Correlation"])
    
    with viz_tabs[0]:  # Pairplot
        st.markdown("<p>Pairplot shows relationships between multiple features colored by species.</p>", unsafe_allow_html=True)
        with st.spinner('Generating pairplot...'):
            fig = plt.figure(figsize=(10, 10))
            sns.pairplot(iris_model.df, hue='species_name', palette=['#FF9999', '#66B2FF', '#99FF99'])
            plt.suptitle('Pairwise Relationships Between Features', y=1.02, fontsize=16)
            st.pyplot(fig)
    
    with viz_tabs[1]:  # Boxplot
        st.markdown("<p>Boxplots show the distribution of values for each feature by species.</p>", unsafe_allow_html=True)
        feature = st.selectbox("Select Feature for Boxplot", iris_model.feature_names)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='species_name', y=feature, data=iris_model.df, hue='species_name', palette=['#FF9999', '#66B2FF', '#99FF99'], legend=False)
        plt.title(f'Distribution of {feature} by Species', fontsize=16, fontweight='bold')
        plt.xlabel('Species', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.xticks(rotation=0, fontsize=12)
        st.pyplot(fig)
    
    with viz_tabs[2]:  # Violin Plot
        st.markdown("<p>Violin plots show the distribution density of each feature by species.</p>", unsafe_allow_html=True)
        feature = st.selectbox("Select Feature for Violin Plot", iris_model.feature_names)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='species_name', y=feature, data=iris_model.df, hue='species_name', palette=['#FF9999', '#66B2FF', '#99FF99'], inner='quartile', legend=False)
        plt.title(f'Distribution Density of {feature} by Species', fontsize=16, fontweight='bold')
        plt.xlabel('Species', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.xticks(rotation=0, fontsize=12)
        st.pyplot(fig)
    
    with viz_tabs[3]:  # Correlation Heatmap
        st.markdown("<p>Correlation heatmap shows how strongly features are related to each other.</p>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = iris_model.X.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Create a mask for the upper triangle
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap with better styling
        sns.heatmap(
            corr, 
            mask=mask,  # Only show the lower triangle
            annot=True,  # Show correlation values
            fmt=".2f",  # Format to 2 decimal places
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Correlation Heatmap of Features', fontsize=16, fontweight='bold')
        st.pyplot(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Model Performance
with tab3:
    st.markdown("<h2 class='sub-header'>Model Performance Comparison</h2>", unsafe_allow_html=True)
    
    # Create a card for overall performance
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Model Accuracy Ranking</h3>", unsafe_allow_html=True)
    st.markdown("<p>Compare how well different machine learning models perform on the iris dataset:</p>", unsafe_allow_html=True)
    
    # Create a dataframe with model accuracies
    accuracies = {name: perf['accuracy'] for name, perf in iris_model.model_performances.items()}
    acc_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy'])
    acc_df = acc_df.sort_values('Accuracy', ascending=False)
    
    # Create a visually enhanced accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a color gradient based on accuracy values
    from matplotlib import colormaps  # Import colormaps
    
    cmap = colormaps.get_cmap('RdYlGn')  # Use colormaps.get_cmap()
    min_acc = acc_df['Accuracy'].min()
    max_acc = acc_df['Accuracy'].max()
    normalized_acc = [(x - min_acc) / (max_acc - min_acc) for x in acc_df['Accuracy']]
    colors = [cmap(x) for x in normalized_acc]
    
    bars = ax.barh(acc_df.index, acc_df['Accuracy'], color=colors)
    
    # Add percentage annotations to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.1%}",
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    # Set chart properties
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Accuracy Score', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    top_accuracy = acc_df['Accuracy'].iloc[0]  # Use .iloc for positional indexing
    top_model = acc_df.index[0]
    top_accuracy = acc_df['Accuracy'].iloc[0]
    
    st.markdown(f"""
    <div style='background-color: #e6ffe6; padding: 1rem; border-radius: 10px; border-left: 5px solid #4CAF50; margin-top: 1rem;'>
        <h3 style='margin: 0; color: #4CAF50;'>Best Performing Model</h3>
        <p style='font-size: 1.2rem; margin: 0.5rem 0;'><span style='font-weight: bold;'>{top_model}</span> with <span style='font-weight: bold;'>{top_accuracy:.1%}</span> accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a table with all accuracies with percentage formatting
    st.markdown("<h3 class='sub-header' style='margin-top: 1.5rem;'>Accuracy Scores</h3>", unsafe_allow_html=True)
    
    # Format percentages for better display
    formatted_acc_df = acc_df.copy()
    formatted_acc_df['Accuracy'] = formatted_acc_df['Accuracy'].apply(lambda x: f"{x:.2%}")
    st.dataframe(formatted_acc_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed metrics card
    st.markdown("<div class='card' style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Detailed Model Performance</h3>", unsafe_allow_html=True)
    st.markdown("<p>Select a model to view its detailed performance metrics:</p>", unsafe_allow_html=True)
    
    # Model selection with enhanced styling
    selected_model = st.selectbox(
        "",
        list(iris_model.models.keys()),
        index=list(iris_model.models.keys()).index(top_model) if top_model in iris_model.models else 0,
        key="metrics_model_selector"
    )
    
    # Get the selected model's performance
    model_perf = iris_model.get_model_performance(selected_model)
    
    # Create two columns for report and confusion matrix
    metrics_col1, metrics_col2 = st.columns([1, 1])
    
    with metrics_col1:
        # Display classification report with enhanced styling
        st.markdown("<h4 style='color: #4B4BFF; margin-bottom: 0.5rem;'>Classification Report</h4>", unsafe_allow_html=True)
        st.markdown("<p>Performance metrics for each species:</p>", unsafe_allow_html=True)
        
        # Process report data
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
        
        # Format the values for better display
        report_df = report_df.round(2)
        
        # Style the dataframe
        st.dataframe(report_df.style.background_gradient(subset=['precision', 'recall', 'f1-score'], cmap="RdYlGn"),
                     use_container_width=True)
        
        # Add explanation of metrics
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem; font-size: 0.9rem;'>
            <h5 style='color: #555; margin: 0 0 0.5rem 0;'>Metrics Explained:</h5>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li><strong>Precision</strong>: When the model predicts a species, how often is it correct?</li>
                <li><strong>Recall</strong>: Of all actual instances of a species, how many did the model identify?</li>
                <li><strong>F1-score</strong>: Harmonic mean of precision and recall.</li>
                <li><strong>Support</strong>: Number of samples of each species in the test dataset.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        # Display confusion matrix with enhanced styling
        st.markdown("<h4 style='color: #4B4BFF; margin-bottom: 0.5rem;'>Confusion Matrix</h4>", unsafe_allow_html=True)
        st.markdown("<p>Shows predicted vs actual species counts:</p>", unsafe_allow_html=True)
        
        # Get and display the confusion matrix
        cm = model_perf['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a prettier heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=iris_model.target_names, 
            yticklabels=iris_model.target_names,
            linewidths=1,
            linecolor='white',
            cbar=False
        )
        
        # Improve the plot aesthetics
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix - {selected_model}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add explanation of confusion matrix
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem; font-size: 0.9rem;'>
            <h5 style='color: #555; margin: 0 0 0.5rem 0;'>Reading the Confusion Matrix:</h5>
            <ul style='margin: 0; padding-left: 1.5rem;'>
                <li>Numbers along the diagonal (top-left to bottom-right) represent <strong>correct predictions</strong>.</li>
                <li>Off-diagonal numbers represent <strong>misclassifications</strong>.</li>
                <li>Higher values along the diagonal and lower values elsewhere indicate better model performance.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance section for this model
    st.markdown("<h3 class='sub-header' style='margin-top: 1.5rem;'>Feature Importance</h3>", unsafe_allow_html=True)
    st.markdown("<p>Which measurements are most important for classification with this model?</p>", unsafe_allow_html=True)
    
    try:
        # Get feature importance for the selected model
        feature_importance = iris_model.get_feature_importance(selected_model)
        
        # Create a more attractive chart for feature importance
        fig, ax = plt.subplots(figsize=(10, 5))
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importance = [importance[i] for i in sorted_indices]
        
        # Create a color gradient
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_features)))
        
        # Create horizontal bars for better readability
        bars = ax.barh(sorted_features, sorted_importance, color=colors)
        
        # Add labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + max(sorted_importance) * 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Improve the plot aesthetics
        ax.set_xlabel('Relative Importance', fontsize=12)
        ax.set_title(f'Feature Importance - {selected_model}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Create a table with percentage contribution
        importance_df = pd.DataFrame({
            'Feature': sorted_features,
            'Importance': sorted_importance,
            'Percentage': [i/sum(sorted_importance)*100 for i in sorted_importance]
        })
        
        # Format the values for better display
        importance_df['Importance'] = importance_df['Importance'].round(3)
        importance_df['Percentage'] = importance_df['Percentage'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(importance_df, use_container_width=True)
        
    except Exception as e:
        st.info(f"Feature importance calculation is not available for {selected_model}. Try another model like Random Forest or Decision Tree.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 4: About Iris Species
with tab4:
    st.markdown("<h2 class='sub-header'>About Iris Species</h2>", unsafe_allow_html=True)
    
    # Create card for the dataset information
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Introduction with enhanced styling
    st.markdown("""
    <h3 style='color: #4B4BFF; margin-bottom: 1rem;'>The Iris Flower Dataset</h3>
    
    <p style='margin-bottom: 1rem;'>
    The Iris flower dataset is one of the most famous datasets in pattern recognition and machine learning.
    It contains measurements for 150 iris flowers from three different species:
    </p>
    """, unsafe_allow_html=True)
    
    # Create beautiful species cards with images
    species_col1, species_col2, species_col3 = st.columns(3)
    
    with species_col1:
        st.markdown("""
        <div style='background-color: #FF99992a; padding: 1rem; border-radius: 10px; height: 100%; text-align: center;'>
            <h4 style='color: #990000; margin-bottom: 0.5rem;'>Iris Setosa</h4>
            <p style='font-size: 0.9rem; text-align: left;'>
            Known for its distinctive appearance with narrow, linear petals and sepals. 
            Typically the smallest of the three species.
            </p>
            <p style='background-color: #FF9999; height: 20px; margin: 0.5rem 0; border-radius: 5px;'></p>
        </div>
        """, unsafe_allow_html=True)
        
    with species_col2:
        st.markdown("""
        <div style='background-color: #66B2FF2a; padding: 1rem; border-radius: 10px; height: 100%; text-align: center;'>
            <h4 style='color: #000099; margin-bottom: 0.5rem;'>Iris Versicolor</h4>
            <p style='font-size: 0.9rem; text-align: left;'>
            Has more oval-shaped petals and is generally larger than Setosa.
            Often referred to as the "Blue Flag" iris.
            </p>
            <p style='background-color: #66B2FF; height: 20px; margin: 0.5rem 0; border-radius: 5px;'></p>
        </div>
        """, unsafe_allow_html=True)
        
    with species_col3:
        st.markdown("""
        <div style='background-color: #99FF992a; padding: 1rem; border-radius: 10px; height: 100%; text-align: center;'>
            <h4 style='color: #009900; margin-bottom: 0.5rem;'>Iris Virginica</h4>
            <p style='font-size: 0.9rem; text-align: left;'>
            Typically the largest of the three species with broad petals.
            Can grow up to 3 feet tall in the wild.
            </p>
            <p style='background-color: #99FF99; height: 20px; margin: 0.5rem 0; border-radius: 5px;'></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section with enhanced styling
    st.markdown("""
    <h3 style='color: #4B4BFF; margin: 1.5rem 0 1rem 0;'>Features in the Dataset</h3>
    
    <p style='margin-bottom: 1rem;'>
    The dataset includes four features measured from each flower:
    </p>
    """, unsafe_allow_html=True)
    
    # Create feature cards with illustrations
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4 style='color: #FF6B6B; margin: 0;'>Sepal Length & Width</h4>
            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
            The <strong>sepal</strong> is the outer part of the flower that protects the bud before it blooms.
            The length and width of sepals are measured in centimeters.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4 style='color: #59C9A5; margin: 0;'>Petal Length & Width</h4>
            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
            The <strong>petal</strong> is the colorful part of the flower that attracts pollinators.
            The length and width of petals are measured in centimeters.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Importance section with enhanced styling
    st.markdown("""
    <h3 style='color: #4B4BFF; margin: 1.5rem 0 1rem 0;'>Importance of the Dataset</h3>
    
    <p style='margin-bottom: 1rem;'>
    The Iris dataset is widely used in machine learning as a beginner's dataset for the following reasons:
    </p>
    
    <ul style='margin-bottom: 1.5rem;'>
        <li><strong>Size:</strong> It's relatively small and easy to understand (150 samples)</li>
        <li><strong>Quality:</strong> It's well-structured and has no missing values</li>
        <li><strong>Simplicity:</strong> It involves a straightforward classification task</li>
        <li><strong>Patterns:</strong> The classes have distinct patterns, making it good for teaching classification algorithms</li>
    </ul>
    
    <h3 style='color: #4B4BFF; margin: 1.5rem 0 1rem 0;'>Historical Significance</h3>
    
    <div style='background-color: #f5f5f5; padding: 1rem; border-radius: 10px; border-left: 4px solid #4B4BFF; margin-bottom: 1.5rem;'>
        <p style='margin: 0;'>
        <strong>Published:</strong> 1936 by British statistician and biologist Ronald Fisher<br>
        <strong>Purpose:</strong> Used to demonstrate the technique of discriminant analysis<br>
        <strong>Legacy:</strong> One of the most widely used datasets in pattern recognition literature
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visual comparison section with enhanced styling
    st.markdown("<div class='card' style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Visual Comparison of Iris Species</h3>", unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    species_viz_tabs = st.tabs(["Average Measurements", "Species Differentiation", "Feature Distribution"])
    
    with species_viz_tabs[0]:
        st.markdown("<p>Compare the average measurements of each feature across the three iris species:</p>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group data by species
        species_data = {}
        for species in iris_model.target_names:
            species_data[species] = iris_model.df[iris_model.df['species_name'] == species]
        
        # Plot average values for each feature by species with improved styling
        width = 0.25
        x = np.arange(len(iris_model.feature_names))
        species_colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        for i, (species, data) in enumerate(species_data.items()):
            means = data[iris_model.feature_names].mean().values
            bars = ax.bar(x + i*width, means, width, label=species, color=species_colors[i], 
                    edgecolor='white', linewidth=1)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Improve the plot aesthetics
        ax.set_xticks(x + width)
        ax.set_xticklabels(iris_model.feature_names, rotation=45, ha='right', fontsize=12)
        ax.set_ylabel('Average Value (cm)', fontsize=12)
        ax.set_title('Average Measurements by Species', fontsize=16, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with species_viz_tabs[1]:
        st.markdown("<p>This visualization shows how the species can be differentiated based on petal and sepal measurements:</p>", unsafe_allow_html=True)
        
        # Create a scatter plot that shows the separation between species
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each species with different colors
        species_colors = {'setosa': '#FF9999', 'versicolor': '#66B2FF', 'virginica': '#99FF99'}
        markers = {'setosa': 'o', 'versicolor': 's', 'virginica': '^'}
        
        for species in iris_model.target_names:
            species_data = iris_model.df[iris_model.df['species_name'] == species]
            ax.scatter(
                species_data['petal length (cm)'], 
                species_data['petal width (cm)'], 
                c=species_colors.get(species, 'gray'),
                label=species,
                s=80,
                marker=markers.get(species, 'o'),
                edgecolor='white',
                alpha=0.8
            )
        
        # Customize the plot
        ax.set_title('Iris Species Differentiation', fontsize=16, fontweight='bold')
        ax.set_xlabel('Petal Length (cm)', fontsize=14)
        ax.set_ylabel('Petal Width (cm)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Species', fontsize=12, title_fontsize=14)
        
        st.pyplot(fig)
        
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem; font-size: 0.9rem;'>
            <p style='margin: 0;'>
            <strong>Key Insight:</strong> Notice how Iris Setosa (red) is clearly separated from the other two species,
            while Versicolor (blue) and Virginica (green) have some overlap. This makes Setosa easy to classify, while
            distinguishing between Versicolor and Virginica is more challenging.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with species_viz_tabs[2]:
        st.markdown("<p>Compare the distribution of measurements across species with violin plots:</p>", unsafe_allow_html=True)
        
        # Create a more informative visualization showing the distribution of all features
        selected_feature = st.selectbox(
            "Select feature to visualize:",
            iris_model.feature_names,
            key="distribution_feature_selector"
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a violin plot with inner box plot
        sns.violinplot(
            x='species_name', 
            y=selected_feature, 
            data=iris_model.df, 
            palette=['#FF9999', '#66B2FF', '#99FF99'],
            inner='quartile',  # Show quartile information inside
            ax=ax
        )
        
        # Add individual points for better visibility
        sns.stripplot(
            x='species_name', 
            y=selected_feature, 
            data=iris_model.df,
            color='black',
            size=4,
            alpha=0.4,
            jitter=True,
            ax=ax
        )
        
        # Customize the plot
        ax.set_title(f'Distribution of {selected_feature} by Species', fontsize=16, fontweight='bold')
        ax.set_xlabel('Species', fontsize=14)
        ax.set_ylabel(selected_feature, fontsize=14)
        
        st.pyplot(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Enhanced sidebar styling and content
st.sidebar.markdown("<h2 style='text-align: center; color: #FF4B4B; margin-bottom: 1rem;'>Iris Predictor</h2>", unsafe_allow_html=True)

# Add a decorative image/separator
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 1.5rem;'>
    <div style='height: 4px; background: linear-gradient(90deg, #FF9999, #66B2FF, #99FF99); border-radius: 2px;'></div>
</div>
""", unsafe_allow_html=True)

# About section with enhanced styling
st.sidebar.markdown("<h3 style='color: #4B4BFF; font-size: 1.2rem;'>🔍 About</h3>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;'>
    <p style='margin: 0; font-size: 0.9rem;'>
    This interactive app uses machine learning to predict iris flower species from measurements. 
    Explore different ML models and visualization tools to understand the famous Iris dataset.
    </p>
</div>
""", unsafe_allow_html=True)

# Instructions with enhanced styling
st.sidebar.markdown("<h3 style='color: #4B4BFF; font-size: 1.2rem;'>📋 How to Use</h3>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;'>
    <ol style='margin: 0; padding-left: 1.2rem; font-size: 0.9rem;'>
        <li>Go to the <strong>Prediction</strong> tab</li>
        <li>Adjust the sliders to input flower measurements</li>
        <li>Select a machine learning model</li>
        <li>Click "Predict Species" to see the result</li>
        <li>Explore other tabs to learn more</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Model descriptions with enhanced styling
st.sidebar.markdown("<h3 style='color: #4B4BFF; font-size: 1.2rem;'>🤖 Models Available</h3>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; font-size: 0.9rem;'>
    <p style='margin-bottom: 0.7rem;'><span style='color: #FF4B4B; font-weight: bold;'>Logistic Regression:</span> A linear model for classification</p>
    <p style='margin-bottom: 0.7rem;'><span style='color: #FF4B4B; font-weight: bold;'>K-Nearest Neighbors:</span> Classification based on closest samples</p>
    <p style='margin-bottom: 0.7rem;'><span style='color: #FF4B4B; font-weight: bold;'>Support Vector Machine:</span> Finds optimal decision boundaries</p>
    <p style='margin-bottom: 0.7rem;'><span style='color: #FF4B4B; font-weight: bold;'>Decision Tree:</span> Uses a tree-like model of decisions</p>
    <p style='margin-bottom: 0.7rem;'><span style='color: #FF4B4B; font-weight: bold;'>Random Forest:</span> Ensemble of decision trees</p>
    <p style='margin-bottom: 0;'><span style='color: #FF4B4B; font-weight: bold;'>Naive Bayes:</span> Probabilistic classifier using Bayes' theorem</p>
</div>
""", unsafe_allow_html=True)

# Add created by info
st.sidebar.markdown("""
<div style='text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f0f0f0; border-radius: 8px;'>
    <p style='margin: 0; font-size: 0.8rem; color: #666;'>
        <strong>Created with</strong><br>
        💻 Python & Streamlit<br>
        🧠 Scikit-learn<br>
        📊 Matplotlib & Seaborn
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    /* General background and text colors */
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .main-title {
        color: #BB86FC;
    }
    .section-title {
        color: #03DAC6;
    }
    .info-box {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    .highlight-text {
        color: #BB86FC;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #03DAC6;
        color: #121212;
        font-weight: bold;
        border: none;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #BB86FC;
        color: #121212;
    }
    .stDataFrame {
        background-color: #1E1E1E;
        color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #03DAC6;
        color: #121212;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #03DAC6;
        color: #121212;
    }
    /* Adjust text and background colors for better readability */
    body {
        background-color: #f5f5f5;
        color: #333333;
    }
    .main-title {
        color: #FF4B4B;
    }
    .section-title {
        color: #4B4BFF;
    }
    .info-box {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #dddddd;
    }
    .highlight-text {
        color: #FF4B4B;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stDataFrame {
        background-color: #ffffff;
        color: #333333;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        color: #333333;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f0f0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
            
</style>
""", unsafe_allow_html=True)

