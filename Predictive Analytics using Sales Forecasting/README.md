# Predictive Analytics for Sales Forecasting

This repository contains my MBA Business Analytics project titled **“Predictive Analytics for Sales Forecasting”**, demonstrating how advanced analytics and machine learning can be applied to retail sales data for accurate demand forecasting and business decision-making.

The project uses the **Superstore Sales Dataset** (`train.csv`), performing data preprocessing, feature engineering, and model-based forecasting using **ARIMA**, **Random Forest**, and **XGBoost**.

---

## 📘 Project Overview

Accurate sales forecasting enables better inventory control, pricing strategy, and supply-chain optimization.  
Traditional forecasting models struggle to adapt to the dynamic nature of retail demand.  
This project leverages **predictive analytics** to build a data-driven forecasting pipeline capable of learning seasonal, regional, and product-level patterns.

---

## 🎯 Objectives

1. Analyze multi-year retail sales data to identify seasonal and cyclical trends.  
2. Engineer predictive features such as lagged sales and rolling averages.  
3. Train forecasting models (ARIMA, Random Forest, XGBoost) to predict future sales.  
4. Evaluate and compare model performance using RMSE, MAE, and R² metrics.  
5. Provide actionable insights to improve demand planning and business forecasting.

---

## 🧠 Methodology

The project follows a structured **predictive analytics pipeline**:

```
Data Collection → Preprocessing → Feature Engineering → Model Training → Forecast Simulation → Evaluation
```

### 🔹 Feature Engineering
```python
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year
df['Sales Lag1'] = df.groupby(['Category'])['Sales'].shift(1)
df['Sales MA3'] = df.groupby(['Category'])['Sales'].transform(lambda x: x.rolling(3).mean())
```

### 🔹 Models Used
- **ARIMA** – Classical time-series model for sequential forecasting  
- **Random Forest Regressor** – Captures nonlinear relationships between features and sales  
- **XGBoost Regressor** – Gradient-boosted model providing strong predictive accuracy  

---

## 🧰 Tools and Libraries

| Category | Tools Used |
|-----------|-------------|
| Programming | Python (Google Colab) |
| Libraries | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, pmdarima |
| Modeling | ARIMA, Random Forest, XGBoost |
| Visualization | Seaborn, Matplotlib |
| Environment | Google Colab |

---

## 📊 Dataset Details

**File Name:** `train.csv`  
**Records:** ~9,900 rows  
**Columns:**  

| Column | Description |
|--------|--------------|
| `Order ID` | Unique order identifier |
| `Order Date` | Date of order placement |
| `Ship Mode` | Shipping method |
| `Customer Name` | Name of customer |
| `Segment` | Market segment (Consumer, Corporate, Home Office) |
| `Country`, `Region`, `City`, `State` | Geographical data |
| `Category`, `Sub-Category` | Product classification |
| `Product Name` | Item sold |
| `Sales` | Sales amount (target variable) |

---

## 📈 Key Visuals

- **Monthly Sales Trend** – Overall growth and seasonality  
- **Regional Sales Heatmap** – Regional demand distribution  
- **Rolling Average vs Actual Sales** – Smoothing of short-term volatility  
- **Category-wise Sales Distribution** – Comparison of product segments  
- **Forecast Simulation** – Predicted vs actual sales for test data  

---

## 💡 Key Insights

- **Technology** and **Furniture** show periodic high-value spikes.  
- **Office Supplies** maintain steady but low-margin sales.  
- **Western region** demonstrates the strongest overall performance.  
- Rolling averages effectively smoothen noise and highlight demand cycles.  
- **XGBoost** achieved the best predictive accuracy among tested models.

---

## 🚀 Future Enhancements

1. Extend model to include **discounts, profit, and shipping cost** as predictors.  
2. Develop **automated forecast dashboards** using Streamlit or Power BI.  
3. Integrate **real-time forecasting** with APIs for continuous updates.  
4. Experiment with **Prophet and LSTM** models for improved temporal learning.  
5. Build a **forecast explainability dashboard** using SHAP or LIME.

---

## 📂 Repository Structure

```
📁 Predictive_Analytics_Sales_Forecasting
│
├── 📘 Akshay_Chavan_Project_Report.docx         # Full MBA project report
├── 📊 Sales_Forecasting_Presentation.pdf        # 10-slide academic presentation
│
├── 📁 data/
│   └── train.csv                                # Superstore Sales dataset
│
├── 📁 visuals/
│   ├── monthly_sales_by_category.png
│   ├── monthly_sales_by_region.png
│   ├── monthly sales trend overall.png
│   ├── regional_category_heatmap.png
│   ├── rolling_average_sales.png
│   └── sales_distribution_by_category.png
│
├── 📁 notebooks/
│   └── forecasting_model.ipynb                  # Python notebook for preprocessing & models
│
└── README.md
```

---

## 📚 References

- Superstore Dataset (Kaggle)  
- Brownlee, J. (2020). Time Series Forecasting with Python: *How to Prepare Data and Develop Models to Predict the Future. Machine Learning Mastery*
- Geron, A. (2019): *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd ed.).*
- Hyndman & Athanasopoulos – *Forecasting: Principles and Practice* 
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018): *"Statistical and Machine Learning Forecasting Methods: Concerns and Ways Forward.*
- Bandara, K., Bergmeir, C., & Hewamalage, H. (2020). *“LSTM-Based Encoder–Decoder for Multi-Step Forecasting: The Role of Exogenous Variables.”* 
- Scikit-learn & XGBoost Documentation, Seaborn and Matplotlib Libraries, Pandas Python Library
- Python Data Science Handbook – Jake VanderPlas  

---

## 👤 Author

**Akshay Chavan**  
MBA – Business Analytics (Batch 2023–2025)  
📍 [LinkedIn](https://www.linkedin.com/in/akshaychavan9/) | [GitHub](https://github.com/AkshayChavan9/)  
📧 akshaychavan678.ac@gmail.com  
