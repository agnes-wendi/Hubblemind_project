# Hubblemind_project

Below is a README file for the provided Python code, which performs data preprocessing, exploratory data analysis (EDA), visualization, and machine learning on an obesity dataset. The README is designed to be clear, concise, and informative, providing an overview of the project, its objectives, and instructions for running the code.
Obesity Dataset Analysis and Classification
Overview
This project analyzes the Obesity Dataset to explore factors influencing obesity levels and predict obesity categories using machine learning models. The dataset contains demographic, lifestyle, and health-related features, with the target variable NObeyesdad indicating obesity levels (e.g., Normal Weight, Obesity Type I, etc.).
The code includes:
Data Preprocessing: Handling missing values, encoding categorical variables, capping outliers, and normalizing continuous features.
Exploratory Data Analysis (EDA): Summary statistics, distribution analysis, correlation analysis, and visualizations (boxplots, histograms, heatmaps, pair plots).
Machine Learning: Training and evaluating Logistic Regression and Random Forest models to predict obesity levels.
Feature Importance: Identifying key features influencing obesity predictions.
Model Evaluation: Assessing model performance using accuracy, classification reports, and confusion matrices.
Dataset
The dataset (ObesityDataSet_raw_and_data_sinthetic.csv) contains 2,111 entries and 17 features, including:
Demographic: Gender, Age, Height, Weight
Lifestyle: Family history of overweight, Frequent consumption of high-calorie food (FAVC), Frequency of vegetable consumption (FCVC), Number of main meals (NCP), Consumption of food between meals (CAEC), Smoking status (SMOKE), Water consumption (CH2O), Calorie monitoring (SCC), Physical activity frequency (FAF), Time using technology (TUE), Alcohol consumption (CALC), Mode of transportation (MTRANS)
Target: Obesity level (NObeyesdad), with categories like Normal Weight, Overweight Level I/II, Obesity Type I/II/III
Prerequisites
To run the code, you need the following installed:
Python 3.7+
Libraries:
pandas
numpy
scikit-learn
seaborn
matplotlib
Install dependencies using:
bash
pip install pandas numpy scikit-learn seaborn matplotlib
Project Structure
ObesityDataSet_raw_and_data_sinthetic.csv: Input dataset (ensure it is placed in the specified directory).
obesity_analysis.py: Main Python script containing all preprocessing, EDA, visualization, and ML code.
README.md: This file, providing project documentation.
How to Run
Prepare the Dataset:
Place the ObesityDataSet_raw_and_data_sinthetic.csv file in the directory specified in the code (e.g., C:\\Users\\wendi\\Downloads\\).
Update the file path in the code if necessary:
python
Obesity_dataset = pd.read_csv("path_to_your_file/ObesityDataSet_raw_and_data_sinthetic.csv")
Run the Code:
Execute the script in a Python environment:
bash
python obesity_analysis.py
The script will:
Load and preprocess the dataset.
Perform EDA and generate visualizations (boxplots, histograms, heatmaps, pair plots).
Train and evaluate Logistic Regression and Random Forest models.
Output model performance metrics and feature importance.
View Outputs:
Visualizations will be displayed as plots (e.g., boxplots, heatmaps, pair plots).
Console outputs include summary statistics, correlation matrices, model accuracy, classification reports, and feature importance.
Code Workflow
Data Loading and Inspection:
Load the dataset using pandas.
Display the first 10 rows, check data types, and verify no missing values.
Data Preprocessing:
Encoding:
Binary variables (e.g., Gender, SMOKE) are encoded using LabelEncoder.
Multi-class variables (e.g., CAEC, CALC, MTRANS, NObeyesdad) are one-hot encoded using pd.get_dummies.
Outlier Handling:
Detect outliers in continuous variables (Age, Height, Weight, NCP) using boxplots.
Cap outliers using the IQR method (values below Q1 - 1.5IQR or above Q3 + 1.5IQR are capped).
Normalization:
Scale continuous variables (Weight, Height, NCP, Age, CH2O, FAF, TUE, FCVC) to [0,1] using MinMaxScaler.
Exploratory Data Analysis (EDA):
Compute summary statistics (describe()).
Analyze distributions of continuous variables using histograms with KDE.
Reverse one-hot encoding for NObeyesdad to visualize relationships.
Create boxplots to explore relationships between continuous variables and obesity levels.
Compute and visualize a correlation matrix for continuous variables using a heatmap.
Visualizations:
Pair Plots: Show pairwise relationships between continuous variables, colored by obesity level.
Feature Importance: Bar plot of feature importance scores from Random Forest.
Confusion Matrix: Heatmap of the confusion matrix for model predictions.
Machine Learning:
Split data into training (80%) and test (20%) sets.
Train two models:
Logistic Regression: Configured with max_iter=1000.
Random Forest: Configured with random_state=42.
Evaluate models using accuracy and classification reports (precision, recall, F1-score).
Note: Both models achieve 100% accuracy, suggesting possible overfitting or an overly clean dataset.
Feature Importance:
Use Random Forest to compute and visualize feature importance scores.
Results
Preprocessing:
No missing values were found.
Outliers in Age, Height, Weight, and NCP were capped successfully.
Continuous variables were normalized to [0,1].
EDA:
Weight and Height show moderate correlation (0.46).
Age and TUE (technology use) have a negative correlation (-0.29).
Boxplots reveal distinct distributions of Weight and Height across obesity levels.
Model Performance:
Logistic Regression: 100% accuracy (potentially due to dataset characteristics).
Random Forest: 100% accuracy (suggests need for further validation).
Feature Importance: Weight, Height, and Age are among the top predictors.
Visualizations:
Pair plots highlight clear separation of obesity categories based on Weight and Height.
Confusion matrix heatmaps show perfect classification (no misclassifications).
Notes
The perfect accuracy (100%) for both models is unusual and may indicate:
An overly clean or synthetic dataset.
Data leakage (e.g., target variable influencing features).
Overfitting due to insufficient model regularization.
To improve robustness:
Apply cross-validation (cross_val_score).
Test on an external dataset.
Add regularization (e.g., C parameter in Logistic Regression, max_depth in Random Forest).
Some visualizations (e.g., pair plots) generate warnings about zero variance, which may require further investigation.
The dataset path is hardcoded; update it to match your local environment.
Future Improvements
Perform cross-validation to validate model performance.
Experiment with additional models (e.g., SVM, XGBoost).
Conduct feature selection to reduce dimensionality.
Explore interactions between features (e.g., Weight * Height for BMI).
Validate results on a separate dataset to ensure generalizability.
