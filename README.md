# 🩺 Obesity Levels Analysis & Prediction

## 📌 Overview
This project focuses on **analyzing obesity levels** and **predicting obesity categories** using machine learning.  
It also includes a simple **FastAPI backend** connected to a **HTML/CSS/JavaScript frontend** for user interaction,  
plus a **dashboard for data analysis**.

---

## 🚀 Project Steps

### 1️⃣ Data Exploration & Understanding
- Loaded the dataset and performed **exploratory data analysis**.
- Checked for missing values, outliers, and class distribution.
- Gained insights into **lifestyle habits** and their relation to obesity.

### 2️⃣ Data Preprocessing
- Applied **PowerTransformer** to some skewed numeric columns.
- Encoded categorical features using **OneHotEncoder**.
- Standardized numerical features with **StandardScaler**.

### 3️⃣ Model Training
- Tested multiple classification models:  
  - Logistic Regression  
  - Decision Trees  
  - Random Forest  
  - Support Vector Machine (SVM)  
- Selected **SVM** for its **best generalization** performance.

### 4️⃣ Clustering Analysis
- Applied clustering to identify **hidden patterns** or possible **sub-classes** in obesity levels.
- Used **KMeans** and visualized clusters for better understanding.

### 5️⃣ API Development (FastAPI)
- Built a **`/predict`** endpoint to return obesity level predictions.
- Saved preprocessing pipelines and model to ensure **consistent predictions**.
- Created a **root endpoint** to serve the frontend HTML page.

### 6️⃣ Frontend Integration
- Designed a **simple HTML form** for input fields:
  - Gender, Age, Height, Weight, Family History, etc.
- Added CSS for styling and form alignment.
- Used JavaScript to send data to FastAPI and display the prediction.

### 7️⃣ Data Analysis Dashboard
- Built **two simple dashboards** to visualize obesity trends.
- Showed feature distributions, correlations, and cluster visualizations.

---

## 📊 Obesity Level Categories
Predictions are mapped as:
- `0`: Insufficient Weight  
- `1`: Normal Weight  
- `2`: Obesity Type I  
- `3`: Obesity Type II  
- `4`: Obesity Type III  
- `5`: Overweight Level I  
- `6`: Overweight Level II  

---

## 🛠 Technologies Used
- **Python** (pandas, numpy, scikit-learn, matplotlib, seaborn)
- **FastAPI** (Backend API)
- **HTML, CSS, JavaScript** (Frontend)
- **Jupyter Notebook** (EDA & Model Development)

---
