# Hubblemind_project



# Overview
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
