import streamlit as st
import pandas as pd

def load_model(model_path):
    model_series = pd.read_pickle(model_path)
    return model_series

def load_base_data(data_path):
    data = pd.read_csv(data_path)
    return data

st.title("Preditor de Churn Telco Telecom")
st.markdown("### Ajuste as informações na barra lateral e faça sua predição")

model_path = "models/classifier.pkl"
data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

data = load_base_data(data_path)
model = load_model(model_path)

st.sidebar.header("Insira as Informações do Cliente")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone = st.sidebar.selectbox("Phone Service", list(data['PhoneService'].unique()))
lines = st.sidebar.selectbox("Multiple Lines", list(data['MultipleLines'].unique()))
internet = st.sidebar.selectbox("Internet Service", list(data['InternetService'].unique()))
security = st.sidebar.selectbox("Online Security", list(data['OnlineSecurity'].unique()))
backup = st.sidebar.selectbox("Online Backup", list(data['OnlineBackup'].unique()))
device_pro = st.sidebar.selectbox("Device Protections", list(data['DeviceProtection'].unique()))
tech_support = st.sidebar.selectbox("Tech Support", list(data['TechSupport'].unique()))
streamingtv = st.sidebar.selectbox("Streaming TV", list(data["StreamingTV"].unique()))
streamingmovies = st.sidebar.selectbox("Streaming Movies", list(data["StreamingMovies"].unique()))
contract = st.sidebar.selectbox("Contract", list(data["Contract"].unique()))
paperless = st.sidebar.selectbox("Paperless Billing", list(data["PaperlessBilling"].unique()))
payment = st.sidebar.selectbox("Payment Method", list(data["PaymentMethod"].unique()))
monthly = st.sidebar.slider("MonthlyCharges", data["MonthlyCharges"].min(), data["MonthlyCharges"].max())
total = st.sidebar.slider("Total Charges", data['MonthlyCharges'].min(), 9000.0)
tenure = st.sidebar.slider("Tenure", data['tenure'].min(), data['tenure'].max())

st.sidebar.header("Links")
st.sidebar.markdown("**Repositório Original**: [Github](https://github.com/datalopes1/telco_pred_churn)")
st.sidebar.markdown("**LinkedIn**: [LinkedIn](https://www.linkedin.com/in/andreluizls1/)")
st.sidebar.markdown("**E-mail:** andreluizlcons@gmail.com")

input_features = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone,
    'MultipleLines': lines,
    'InternetService': internet,
    'OnlineSecurity': security,
    'OnlineBackup': backup,
    'DeviceProtection': device_pro,
    'TechSupport': tech_support,
    'StreamingTV': streamingtv,
    'StreamingMovies': streamingmovies,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly,
    'TotalCharges': total

}

input_df = pd.DataFrame([input_features])

with st.container():
    st.write("")  
    
    if st.button("Predict Churn"):
        pred = model['model'].predict_proba(input_df[model['features']])[:, 1]
        prediction = float(pred) * 100

        st.markdown("### Resultado")
        st.write(f"A probabilidade de churn para este cliente é: **:red[{prediction:.2f}%]**")

        if pred > 0.60:
            st.write("De acordo com nosso limiar de decisão, este cliente seria caso de Churn")
            img_path = "doc/img/jPv7c5yLaBe8HoUV0Gbi--3--gjeze.jpg"
            st.image(img_path, use_column_width=True)
        else:
            st.write("De acordo com nosso limiar de decisão, este cliente não seria caso de Churn")
            img_path = "doc/img/q1yjN8qqef7659j5W7ri--3--rdj1l.jpg"
            st.image(img_path, use_column_width=True)
