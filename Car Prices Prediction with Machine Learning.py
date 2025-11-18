# ===========================
# Car Prices Prediction 
# ===========================

# 1️⃣ Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import random

# Optional: Interactive widgets (for Jupyter Notebook)
try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:
    print("⚠️ ipywidgets not installed, interactivity will be disabled.")

# ---------------------------
# 2️⃣ Load Dataset
# ---------------------------
data = pd.DataFrame({
    'Car_Name': ['ritz','sx4','ciaz','wagon r','swift','vitara brezza','ciaz','s cross','ciaz','ciaz'],
    'Year': [2014,2013,2017,2011,2014,2018,2015,2015,2016,2015],
    'Selling_Price':[3.35,4.75,7.25,2.85,4.6,9.25,6.75,6.5,8.75,7.45],
    'Present_Price':[5.59,9.54,9.85,4.15,6.87,9.83,8.12,8.61,8.89,8.92],
    'Driven_kms':[27000,43000,6900,5200,42450,2071,18796,33429,20273,42367],
    'Fuel_Type':['Petrol','Diesel','Petrol','Petrol','Diesel','Diesel','Petrol','Diesel','Diesel','Diesel'],
    'Selling_type':['Dealer','Dealer','Dealer','Dealer','Dealer','Dealer','Dealer','Dealer','Dealer','Dealer'],
    'Transmission':['Manual','Manual','Manual','Manual','Manual','Manual','Manual','Manual','Manual','Manual'],
    'Owner':[0,0,0,0,0,0,0,0,0,0]
})

# ---------------------------
# 3️⃣ Basic Data Info
# ---------------------------
print("Dataset Shape:", data.shape)
print(data.head())
print(data.info())
print(data.describe())

# ---------------------------
# 4️⃣ Preprocessing
# ---------------------------
# Encode categorical variables
label_encoders = {}
categorical_cols = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature & Target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 5️⃣ Model Training
# ---------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------------------
# 6️⃣ Predictions & Evaluation
# ---------------------------
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# ---------------------------
# 7️⃣ Visualization
# ---------------------------
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='Present_Price', y='Selling_Price', data=data, hue='Fuel_Type', palette='Set2', s=100)
plt.title("Present Price vs Selling Price")
plt.show()

# Feature importance
importances = model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance from Random Forest")
plt.show()

# ---------------------------
# 8️⃣ Predict Selling Price for New Input
# ---------------------------
def predict_selling_price(car_name, year, present_price, driven_kms, fuel_type, selling_type, transmission, owner):
    car_name_encoded = label_encoders['Car_Name'].transform([car_name])[0]
    fuel_type_encoded = label_encoders['Fuel_Type'].transform([fuel_type])[0]
    selling_type_encoded = label_encoders['Selling_type'].transform([selling_type])[0]
    transmission_encoded = label_encoders['Transmission'].transform([transmission])[0]
    
    input_data = np.array([[car_name_encoded, year, present_price, driven_kms, fuel_type_encoded,
                            selling_type_encoded, transmission_encoded, owner]])
    
    input_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)[0]
    return round(predicted_price, 2)

print("💰 Example Prediction:", predict_selling_price('ciaz', 2017, 9.85, 6900, 'Petrol', 'Dealer', 'Manual', 0))

# ---------------------------
# 9️⃣ Save Model & Scaler
# ---------------------------
joblib.dump(model, "car_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ Model, scaler, and encoders saved.")

# ---------------------------
# 🔟 Full Dataset Predictions
# ---------------------------
X_scaled = scaler.transform(X)
full_predictions_df = pd.DataFrame({
    'Car_Name': data['Car_Name'],
    'Actual_Price': data['Selling_Price'],
    'Predicted_Price': model.predict(X_scaled)
})
full_predictions_df.to_csv("car_price_full_predictions.csv", index=False)
print("✅ Full dataset predictions saved.")

# ---------------------------
# 1️⃣ Quirky & Engaging Ending
# ---------------------------
quirky_quotes = [
    "🚗💨 This car's price might just outrun your morning coffee!",
    "💸 Money whispers secrets… and it just revealed yours.",
    "😎 Cars don't gossip, but numbers do — listen closely!",
    "⚡ Your neighbor bought a car last week… check our prediction before he brags!",
    "🧙‍♂️ A wizard predicted it… but we double-checked the math!"
]

print("\n✨ Today’s Revelation:")
print(random.choice(quirky_quotes))

# Playful stats summary
print("\n📊 Quick Stats:")
print(f"🛻 Total Cars Analyzed: {len(data)}")
print(f"💎 Highest Price: {data['Selling_Price'].max()} lakhs")
print(f"💸 Average Price: {round(data['Selling_Price'].mean(),2)} lakhs")
most_common_fuel = data['Fuel_Type'].mode()[0]
# Decode fuel type
most_common_fuel_name = label_encoders['Fuel_Type'].inverse_transform([most_common_fuel])[0]
print(f"🛠️ Most Common Fuel Type: {most_common_fuel_name}")

# Random Car Fortune
fortunes = [
    "🔮 A mysterious buyer might appear soon… keep your car polished!",
    "💰 Patience pays: waiting a month could raise the selling price!",
    "🕵️ Someone is secretly checking your car’s value… stay alert!",
    "🌟 Your car dreams are closer than you think… literally!"
]
print("\n💡 Car Fortune:")
print(random.choice(fortunes))

# ASCII Art Car
ascii_car = r"""
      ______
     /|_||_\`.__
    (   _    _ _\
    =`-(_)--(_)-'
"""
print(ascii_car)

# Playful sound effect
sounds = ["Vroom Vroom! 🏎️", "Brmmm… Brmmm… 🚙", "Beeep! 🚗💨", "Zoom Zoom! ⚡"]
print(random.choice(sounds))

# Smooth final closing
print("\n🎯 Script Completed! You are now officially a Car Price Guru 🔮🚗")
print("📈 Predictions done, plots ready, and your code is shining!")
