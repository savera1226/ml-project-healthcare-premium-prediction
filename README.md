🏥 AI Health Insurance Premium Predictor

This is a web application, built with Streamlit, that uses a machine learning model to predict annual health insurance premiums based on user inputs.

🚀 View the Live App

You can access the live, deployed version of this project here:

https://ml-project-healthcare-premium-prediction-2025-krish.streamlit.app/

📋 Features

📈 AI-Powered Predictions: Uses a trained machine learning model (Random Forest/Gradient Boosting) to estimate premium costs.

✨ Modern UI: A clean, polished, and responsive user interface built with Streamlit and custom CSS.

💡 Personalized Insights: Provides actionable recommendations (e.g., "Quit Smoking," "Weight Management") based on the user's risk profile.

📊 Cost Breakdown: Instantly calculates estimated monthly and daily premium costs.

🛡️ Safe Fallback Mode: Includes a robust fallback estimator to ensure the app remains functional even if the primary AI models fail to load.

🛠️ Tech Stack

Python: The core programming language.

Streamlit: For building and deploying the interactive web app.

Pandas: For data manipulation and preprocessing.

Scikit-learn (sklearn): For the machine learning models and scalers.

Joblib: For loading the pre-trained model artifacts.

📖 Project Overview

This project was built as part of a machine learning course. The goal was to build an end-to-end data science application, which involved:

Data Preprocessing: Cleaning and preparing the dataset for training.

Model Training: Experimenting with and training regression models (like Random Forest) to predict insurance charges.

Model Serialization: Saving the trained models and scalers as .joblib artifacts.

App Development: Building a user-friendly frontend with Streamlit to interact with the model.

Deployment: Deploying the final application to Streamlit Cloud for public access.

How to Run This Project Locally

To run this project on your own machine, follow these steps:

Clone the repository:

git clone [https://github.com/savera1226/ml-project-healthcare-premium-prediction.git](https://github.com/savera1226/ml-project-healthcare-premium-prediction.git)
cd ml-project-healthcare-premium-prediction


Install dependencies:
(It's highly recommended to use a virtual environment)

pip install -r requirements.txt


(Note: You will need to create a requirements.txt file listing streamlit, pandas, numpy, and scikit-learn)

Run the app:

streamlit run main.py
