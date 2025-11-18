# 🚗 Car Price Prediction with Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green) ![LinkedIn](https://img.shields.io/badge/LinkedIn-Rao%20Umar-blue?logo=linkedin)

**Author:** Khadija Rao  
**Email:** [emailraoumar0058@gmail.com](mailto:emailraoumar0058@gmail.com)  
**LinkedIn:** [Rao Umar](https://www.linkedin.com/in/rao-umar-904807355)

---

## 🔍 Project Overview

This project predicts the **selling price of used cars in India** using a **Random Forest Regressor**.  
It offers a **dashboard-like experience** in Jupyter Notebook and includes:

- Data preprocessing and feature encoding
- Model training and evaluation
- Visualization of feature importance and correlations
- Predicting new car prices using saved model, scaler, and label encoders
- Fun, interactive messages & quick stats summary

---

## 🧰 Key Features

- **Automatic Data Encoding:** Converts categorical variables to numeric format.
- **Random Forest Model:** High-accuracy predictions on unseen data.
- **Visualizations:** Scatter plots, correlation heatmaps, and feature importance charts.
- **Interactive Sliders (Jupyter Notebook):** Test predictions for any car configuration.
- **Model Persistence:** Save/load model, scaler, and label encoders for future use.
- **CSV Predictions:** Full dataset predictions saved for easy analysis.

---

## 📂 Repository Structure

```text
car-price-prediction/
│
├── Car Prices Prediction with Machine Learning.py   # Main Python script
├── Car Prices Prediction Notebook.ipynb            # Optional Jupyter Notebook
├── car_price_model.pkl                             # Trained Random Forest model
├── scaler.pkl                                      # StandardScaler for features
├── label_encoders.pkl                              # LabelEncoders for categorical data
├── car_price_full_predictions.csv                  # Predictions for full dataset
├── README.md                                       # Project documentation
└── demo.gif                                        # Optional interactive demo
````

---

## 📊 Sample Outputs

### Predicted Selling Price

```text
💰 Predicted Selling Price: 7.6 lakhs
```

### Model Evaluation (Test Set)

```text
Mean Squared Error (MSE): 0.12
R² Score: 0.95
```

### Visualization Examples

* Feature Correlation Heatmap
* Present Price vs Selling Price Scatter Plot
* Feature Importance Bar Chart

---

## 🚀 How to Use

1. **Clone the repository**

```bash
git clone https://github.com/rao-umar/car-price-prediction.git
cd car-price-prediction
```

2. **Install required libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib ipywidgets
```

3. **Run the Python script**

```bash
python "Car Prices Prediction with Machine Learning.py"
```

4. **Optional Jupyter Notebook version**

```bash
jupyter notebook "Car Prices Prediction Notebook.ipynb"
```

5. **Use interactive sliders** to predict car prices.
6. **See fun stats and playful messages** after each prediction.

---

## 💡 Fun & Interactive Features

* Random playful messages after predictions
* Mini ASCII car art:

```text
      ______
     /|_||_\`.__
    (   _    _ _\
    =`-(_)--(_)-'
```

* Quick stats summary:

  * Total cars analyzed
  * Highest price
  * Most common fuel type
* Random sounds & “mysterious fortunes” after prediction

---

## 🏁 Conclusion

This project combines **data science, machine learning, and interactivity** to create a smooth, user-friendly experience.
It can be extended to include more cars, additional features, or different ML models for higher prediction accuracy.

---

## 📧 Contact

**Khadija Rao**
Email: [emailraoumar0058@gmail.com](mailto:emailraoumar0058@gmail.com)
LinkedIn: [Rao Umar](https://www.linkedin.com/in/rao-umar-904807355)

Feel free to connect, collaborate, or share feedback!

