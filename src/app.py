import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from feature_extraction import extract_features
from models import train_svm, train_random_forest, evaluate_model
from PIL import Image

# Set page config and add OM logo in the header
st.set_page_config(page_title=" NeurOM", page_icon="🕉️", layout="wide")

# Add CSS for custom styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("d:/CODING/Data Science/omchant/src/styles.css")

# Additional CSS styling
st.markdown(
    """
    <style>
    .main-title {
        color: #FF5733;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-top: -40px;
    }
    .accuracy-box {
        background-color: #d0f0fd; /* Light blue background */
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #87CEFA;
        margin: 10px;
        color: #003366; /* Dark blue text */
    }
    </style>
    """, unsafe_allow_html=True)

# Header section with OM Logo and Title
st.image("images/omsym.png", width=100,)
st.markdown('<h1 class="main-title">NeurOM: OM Chants and Its Effect on Mind</h1>', unsafe_allow_html=True)

# Load the dataset
data = pd.read_csv('../datasets/eeg_data.csv')

# Extract features and labels
X = data[['skewness', 'variance', 'kurtosis', 'shannon_entropy']]
y = data['label']

# Sidebar controls for parameter adjustment
st.sidebar.header("Adjust Parameters")
skewness_range = st.sidebar.slider("Skewness", float(X['skewness'].min()), float(X['skewness'].max()), (float(X['skewness'].min()), float(X['skewness'].max())))
variance_range = st.sidebar.slider("Variance", float(X['variance'].min()), float(X['variance'].max()), (float(X['variance'].min()), float(X['variance'].max())))
kurtosis_range = st.sidebar.slider("Kurtosis", float(X['kurtosis'].min()), float(X['kurtosis'].max()), (float(X['kurtosis'].min()), float(X['kurtosis'].max())))
entropy_range = st.sidebar.slider("Shannon Entropy", float(X['shannon_entropy'].min()), float(X['shannon_entropy'].max()), (float(X['shannon_entropy'].min()), float(X['shannon_entropy'].max())))

# Filter data based on sidebar input
filtered_data = data[
    (X['skewness'] >= skewness_range[0]) & (X['skewness'] <= skewness_range[1]) &
    (X['variance'] >= variance_range[0]) & (X['variance'] <= variance_range[1]) &
    (X['kurtosis'] >= kurtosis_range[0]) & (X['kurtosis'] <= kurtosis_range[1]) &
    (X['shannon_entropy'] >= entropy_range[0]) & (X['shannon_entropy'] <= entropy_range[1])
]

X_filtered = filtered_data[['skewness', 'variance', 'kurtosis', 'shannon_entropy']]
y_filtered = filtered_data['label']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Train SVM model
svm_model = train_svm(X_train, y_train)

# Train Random Forest model
rf_model = train_random_forest(X_train, y_train)

# Evaluate SVM
svm_accuracy = evaluate_model(svm_model, X_test, y_test)

# Evaluate Random Forest
rf_accuracy = evaluate_model(rf_model, X_test, y_test)

# Display accuracy in a box
st.subheader("Model Accuracy")
st.markdown(f'<div class="accuracy-box"><b>SVM Accuracy:</b> {svm_accuracy * 100:.2f}%</div>', unsafe_allow_html=True)
st.markdown(f'<div class="accuracy-box"><b>Random Forest Accuracy:</b> {rf_accuracy * 100:.2f}%</div>', unsafe_allow_html=True)

# Confusion matrix for both models
st.subheader("Confusion Matrix")

# Confusion Matrix for SVM
st.write("SVM Confusion Matrix:")
svm_conf_matrix = confusion_matrix(y_test, svm_model.predict(X_test))
fig, ax = plt.subplots(figsize=(5, 5))  # Fixed size for better readability
sns.heatmap(svm_conf_matrix, annot=True, fmt='g', cmap='Blues', ax=ax, cbar=False)
st.pyplot(fig)

# Confusion Matrix for Random Forest
st.write("Random Forest Confusion Matrix:")
rf_conf_matrix = confusion_matrix(y_test, rf_model.predict(X_test))
fig, ax = plt.subplots(figsize=(5, 5))  # Fixed size for better readability
sns.heatmap(rf_conf_matrix, annot=True, fmt='g', cmap='Greens', ax=ax, cbar=False)
st.pyplot(fig)

plt.close()
# 'About Us' Section
st.subheader("About Us")
st.write(""" 
We are a team of passionate students from Netaji Subhas University of Technology, currently in our 7th semester pursuing Electronics and Communication Engineering with a specialization in Artificial Intelligence and Machine Learning (ECAM).

Our team members include:
        
Madhav Aggarwal (2021UEA6570)
Vidit Jain (2021UEA6589)
Gurmesh Singh (2021UEA6618)
Nimish Goyal (2021UEA6634)

This project, titled "OM Chants and Its Effects on the Human Mind", is part of our BTech coursework. The primary aim of this project is to analyze how OM mantra meditation impacts brain activity using Electroencephalogram (EEG) data. We utilize advanced machine learning techniques, specifically Support Vector Machines (SVM) and Random Forest, to classify brain states before and after meditation.

Through this project, we hope to demonstrate the calming and positive influence of OM mantra meditation on mental and cognitive processes, quantified through changes in brainwave patterns.

We have been guided and mentored by Dr. Manisha Khulbe, whose expertise and constant support have been invaluable in helping us bring this project to life.

This project reflects our interest in applying cutting-edge AI techniques to real-world problems, particularly those that intersect health, wellness, and technology. We hope our project showcases the potential of meditation as a scientifically measurable tool for enhancing mental well-being.
""")

# Footer with OM logo
st.image("images/omsym.png", width=50)
