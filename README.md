# Time Series Sales Forecasting Web App

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

An end-to-end Time Series Sales Forecasting web application designed to predict the number of products sold based on historical sales data. Built and deployed from scratch, this project encompasses comprehensive data preprocessing, feature engineering, and the training of an Artificial Neural Network (ANN).

**[🔴 Live Demo](https://timeseriesannproject.streamlit.app/)** | **[GitHub Repository](https://github.com/jeswinandrew11-beep/time-series-project)**

### App Walkthrough
<video src="https://github.com/user-attachments/assets/c3dd8cc7-4cc4-4753-a415-992ff0e8189e" width="100%" controls></video>
---

## Dataset

The model is trained on a robust sales dataset containing over 300,000 observations. 

**Features:**
- Date
- Country
- Store
- Product

**Target Variable:**
- Number of Products Sold

---

## Data Preprocessing & Feature Engineering

To ensure high data quality and improve model accuracy, extensive feature engineering was performed before training:

- **Date Extraction:** Extracted Day, Month, Year, Week of Month, Week of Year, and Weekday/Weekend identifiers.
- **Holiday Integration:** Generated country-specific holiday features leveraging the Country and Date columns.
- **Scaling:** Applied Min-Max Scaling to normalize numerical inputs.
- **Encoding:** Utilized One-Hot Encoding and Ordinal Encoding for categorical variables.

---

## Model Development & Performance

The core prediction engine is an **Artificial Neural Network (ANN)** trained on the engineered feature set. 

Here is the breakdown of the model's performance across different evaluation splits:

| Metric | Training | Validation | Test |
| :--- | :--- | :--- | :--- |
| **RMSE** | 145.76 | 116.62 | 134.60 |
| **MAE** | 86.69 | 74.32 | 83.85 |
| **R² Score** | 0.9593 | 0.9646 | 0.9480 |
| **sMAPE** | 29.19% | 27.57% | 27.52% |

---

## Deployment

The trained model is integrated into a fully interactive web application built with **Streamlit**. It is deployed online, allowing users to input parameters and receive real-time sales forecasts.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Git & GitHub

---

## What I Learned

Building this end-to-end Machine Learning solution was a valuable learning experience. It significantly strengthened my practical understanding of:

- Time Series Forecasting
- Feature Engineering
- Data Preprocessing
- Artificial Neural Networks
- Model Evaluation
- Deploying Machine Learning Applications

*Feedback and suggestions from the community are always appreciated! Feel free to open an issue or reach out.*
