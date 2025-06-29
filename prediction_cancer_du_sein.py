import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@st.cache_data
def load_model_scaler():
    df = pd.read_csv('Breast_Cancer.csv')
    df['Status'] = df['Status'].map({'Alive': 0, 'Dead': 1})

    # Colonnes numériques
    num_cols = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']

    # Colonnes catégorielles à one-hot encoder (sans 'Marital Status')
    cat_cols = ['Race', 'T Stage ', 'N Stage', '6th Stage',
                'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']

    X = df[num_cols + cat_cols]
    y = df['Status']

    X_encoded = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, scaler, X_encoded.columns.tolist()


model, scaler, feature_cols = load_model_scaler()

st.title("Prédiction de survie des patients atteint du cancer du sein")

# Inputs numériques
age = st.number_input("Age", min_value=0, max_value=120, value=53)
tumor_size = st.number_input("Tumor Size", min_value=0.0, max_value=100.0, value=30.0)
regional_node_examined = st.number_input("Regional Node Examined", min_value=0, max_value=50, value=14)
regional_node_positive = st.number_input("Reginol Node Positive", min_value=0, max_value=50, value=4)
survival_months = st.number_input("Survival Months", min_value=0, max_value=200, value=71)

# Rechargement pour catégories
df = pd.read_csv('Breast_Cancer.csv')


def select_category(col_name):
    options = sorted(df[col_name].dropna().unique())
    return st.selectbox(f"{col_name}", options)


race = select_category('Race')
t_stage = select_category('T Stage ')
n_stage = select_category('N Stage')
sixth_stage = select_category('6th Stage')
differentiate = select_category('differentiate')
grade = select_category('Grade')
a_stage = select_category('A Stage')
estrogen_status = select_category('Estrogen Status')
progesterone_status = select_category('Progesterone Status')

# Création du dict d'entrée avec initialisation à 0
input_dict = {col: 0 for col in feature_cols}

# Affecter les variables numériques
input_dict['Age'] = age
input_dict['Tumor Size'] = tumor_size
input_dict['Regional Node Examined'] = regional_node_examined
input_dict['Reginol Node Positive'] = regional_node_positive
input_dict['Survival Months'] = survival_months


# Fonction pour encoder les catégories en one-hot (drop_first=True)
def set_dummy(prefix, val):
    col_name = f"{prefix}_{val}"
    if col_name in feature_cols:
        input_dict[col_name] = 1


set_dummy('Race', race)
set_dummy('T Stage ', t_stage)
set_dummy('N Stage', n_stage)
set_dummy('6th Stage', sixth_stage)
set_dummy('differentiate', differentiate)
set_dummy('Grade', grade)
set_dummy('A Stage', a_stage)
set_dummy('Estrogen Status', estrogen_status)
set_dummy('Progesterone Status', progesterone_status)

input_df = pd.DataFrame([input_dict])

# Standardisation
input_scaled = scaler.transform(input_df)

if st.button("Prédire"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    if pred == 1:
        st.error(f"Prédiction : Dead (probabilité = {proba:.2f})")
    else:
        st.success(f"Prédiction : Alive (probabilité = {1 - proba:.2f})")
