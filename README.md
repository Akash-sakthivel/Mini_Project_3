# 🌾 Crop Production Prediction

## Project Overview
The Crop Production Prediction project aims to forecast agricultural production using historical data on crop yields, harvested area, and environmental factors. By leveraging machine learning and Streamlit, this project provides insights to support farmers, policymakers, and agribusinesses in food security planning, supply chain optimization, and market forecasting.

### Objectives
- To clean, preprocess, and structure crop production data.
- To analyze production trends, environmental influences, and regional productivity.
- To develop an interactive Streamlit dashboard for exploration.
- To provide data-driven insights for agricultural planning and policy-making.

## Technologies Used
- **Programming Language**: Python
- **Web Application Framework**: Streamlit
- **Data Visualization**: Matplotlib, Plotly, Seaborn
- **Data Processing**: Pandas, NumPy

## Setup Instructions

### Prerequisites
- Python 3.0 installed on your machine.
- Jupyter Notebook (optional for data exploration).

### Installation
1. **Install Required Packages**:
   ```bash
   pip install streamlit pandas sqlalchemy matplotlib numpy plotly seaborn joblib
   ```
   
2. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```
   
## Project Structure
```
Crop-Production-Prediction/
│── app.py                        # Main Streamlit application
│── Crop_Production_Pred.ipynb    # EDA and prediction model script
│── models/
│   │── crop_production_model.pkl  # Trained machine learning model
│   │── model_features.pkl         # Feature mappings for model input
│── data/
│   │── cleaned_crop_data.csv       # Processed dataset
│── README.md                      # Project documentation
```

## Dataset Description
The dataset consists of multiple Excel sheets, each representing different agricultural units with crop production observations.

### Key Columns
- **Geographical Information**: Area, Area Code
- **Temporal Data**: Year, Year Code
- **Crop Information**: Item (Crop Name), Item Code
- **Production Data**: Area Harvested (ha), Yield (kg/ha), Production (tons)

### Data Cleaning & Preprocessing

- Handling missing values and standardizing data formats.
- One-hot encoding categorical variables (Area, Crop).
- Feature engineering for improved prediction accuracy.

## Data Analysis

### Exploratory Data Analysis (EDA)
1. **Temporal Trends**:
   - Yearly production analysis to track growth and decline.
   - Identifying seasonal variations in yield and area harvested.

2. **Geographical Analysis**:
   - Comparing productivity across different regions.
   - Identifying high-yield vs. low-yield areas.

3. **Production Correlation Analysis**:
   - Analyzing relationships between Area Harvested, Yield, and Production.
   - Assessing environmental and economic influences on production trends.

## Data Visualization & Dashboard

### Interactive Streamlit Dashboard
- Home Page: Overview of crop production trends.
- Crop Distribution: Production comparisons across different crops and regions.
- Agricultural Insights: Trends and patterns in historical production data.
  
### Example Visualizations
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crop Production Analysis
data = pd.read_csv("cleaned_crop_data.csv")
plt.figure(figsize=(10, 6))
sns.lineplot(x="Year", y="Production", hue="Item", data=data)
plt.xlabel("Year")
plt.ylabel("Total Production (tons)")
plt.title("Crop Production Trends Over the Years")
plt.show()
```

## Challenges and Solutions

### Challenges
-  Handling missing data across multiple sheets.
- Ensuring accurate predictions with diverse crop types.
- Optimizing dashboard performance for large datasets.

### Solutions
- Data imputation techniques for missing values.
- Feature selection and transformation to improve model accuracy.
- SQL optimizations and caching for fast dashboard performance.

### Future Enhancements
- Integrate real-time weather data for dynamic yield predictions.
- Implement deep learning models for improved accuracy.
- Add geospatial analysis to map production trends visually.

### Conclusion
The **Crop Production Prediction** project provides valuable insights into agricultural productivity, crop yield trends, and regional variations. Through data cleaning, exploratory analysis, and predictive modeling, this project helps stakeholders in agriculture, policymaking, and agribusiness make informed decisions. The interactive Streamlit dashboard enhance data exploration, making it easier to analyze production trends and forecast future outputs. Future enhancements, such as integrating machine learning for weather-based predictions and geospatial analysis, can further improve the accuracy and impact of crop production forecasting.

### References
- [Streamlit Documentation](https://docs.streamlit.io/)
