import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Path untuk model dan preprocessing tools
MODEL_PATH = "best_model.pkl"
IMPUTER_PATH = "imputer.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
BEST_MODEL_INFO_PATH = "best_model_info.pkl"
DATA_URL = "https://github.com/aguskurniawan10/prediksiNKLabUBPJPR/raw/main/DATA%20PREDIKSI%20NK%20LAB%202025.xlsx"

def train_and_save_model():
    """Melatih model dan menyimpannya menggunakan pickle."""
    df = pd.read_excel(DATA_URL)
    df.columns = df.columns.str.strip()

    required_columns = ['Suppliers', 'GCV ARB UNLOADING', 'TM ARB UNLOADING', 
                        'Ash Content ARB UNLOADING', 'Total Sulphur ARB UNLOADING', 'GCV (ARB) LAB']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset!")

    label_encoder = LabelEncoder()
    df['Suppliers'] = label_encoder.fit_transform(df['Suppliers'])

    X = df[['Suppliers', 'GCV ARB UNLOADING', 'TM ARB UNLOADING', 
            'Ash Content ARB UNLOADING', 'Total Sulphur ARB UNLOADING']]
    y = df['GCV (ARB) LAB']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_numeric = X_train.drop(columns=['Suppliers'])
    X_test_numeric = X_test.drop(columns=['Suppliers'])

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_numeric)
    X_test_imputed = imputer.transform(X_test_numeric)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    X_train_final = np.hstack([X_train[['Suppliers']].values, X_train_scaled])
    X_test_final = np.hstack([X_test[['Suppliers']].values, X_test_scaled])

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    best_model = None
    best_score = float('-inf')
    best_model_name = ""
    
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)
    with open(IMPUTER_PATH, "wb") as file:
        pickle.dump(imputer, file)
    with open(SCALER_PATH, "wb") as file:
        pickle.dump(scaler, file)
    with open(ENCODER_PATH, "wb") as file:
        pickle.dump({"encoder": label_encoder, "classes": label_encoder.classes_}, file)
    with open(BEST_MODEL_INFO_PATH, "wb") as file:
        pickle.dump({"name": best_model_name, "r2": best_score}, file)

if not os.path.exists(BEST_MODEL_INFO_PATH) or not os.path.exists(MODEL_PATH):
    train_and_save_model()

with open(MODEL_PATH, "rb") as file:
    best_model = pickle.load(file)
with open(IMPUTER_PATH, "rb") as file:
    imputer = pickle.load(file)
with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)
with open(ENCODER_PATH, "rb") as file:
    encoder_data = pickle.load(file)
    label_encoder = encoder_data["encoder"]
    label_encoder.classes_ = encoder_data["classes"]
with open(BEST_MODEL_INFO_PATH, "rb") as file:
    best_model_info = pickle.load(file)

st.title("Prediksi GCV (ARB) LAB")
st.write(f"**Model yang digunakan:** {best_model_info['name']} dengan **R² = {best_model_info['r2']:.4f}**")

supplier_options = list(label_encoder.classes_)

supplier1_selected = st.selectbox("Supplier 1", supplier_options, index=0)
supplier2_selected = st.selectbox("Supplier 2", supplier_options, index=1)

percentage_supplier1 = st.slider("Persentase Supplier 1", 0, 100, 50)
percentage_supplier2 = 100 - percentage_supplier1

supplier1_encoded = label_encoder.transform([supplier1_selected])[0]
supplier2_encoded = label_encoder.transform([supplier2_selected])[0]
supplier_avg = (supplier1_encoded * percentage_supplier1 + supplier2_encoded * percentage_supplier2) / 100

def weighted_avg(val1, val2):
    return (val1 * percentage_supplier1 + val2 * percentage_supplier2) / 100

gcv_arb_1 = st.number_input("GCV ARB UNLOADING Supplier 1", value=4200.0)
gcv_arb_2 = st.number_input("GCV ARB UNLOADING Supplier 2", value=4200.0)
gcv_arb_avg = weighted_avg(gcv_arb_1, gcv_arb_2)

tm_arb_1 = st.number_input("TM ARB UNLOADING Supplier 1", value=35.5)
tm_arb_2 = st.number_input("TM ARB UNLOADING Supplier 2", value=35.5)
tm_arb_avg = weighted_avg(tm_arb_1, tm_arb_2)

ash_content_1 = st.number_input("Ash Content ARB UNLOADING Supplier 1", value=5.0)
ash_content_2 = st.number_input("Ash Content ARB UNLOADING Supplier 2", value=5.0)
ash_content_avg = weighted_avg(ash_content_1, ash_content_2)

total_sulphur_1 = st.number_input("Total Sulphur ARB UNLOADING Supplier 1", value=0.3)
total_sulphur_2 = st.number_input("Total Sulphur ARB UNLOADING Supplier 2", value=0.3)
total_sulphur_avg = weighted_avg(total_sulphur_1, total_sulphur_2)

if st.button("Prediksi"):
    input_data = np.array([[gcv_arb_avg, tm_arb_avg, ash_content_avg, total_sulphur_avg]])
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)
    input_final = np.hstack([[supplier_avg], input_scaled[0]])
    prediction = best_model.predict([input_final])
    st.success(f"Prediksi GCV (ARB) LAB: {prediction[0]:.2f}")
