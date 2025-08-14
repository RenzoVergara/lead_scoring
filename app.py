# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
import plotly.express as px

st.set_page_config(page_title="Lead Scoring App", layout="wide")

st.title("📊 Lead Scoring Inteligente")

# --- Subir CSV ---
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Vista previa de datos")
    st.dataframe(df.head())

    # --- Separar features y target ---
    if "Cliente" not in df.columns:
        st.error("❌ El archivo debe contener una columna 'Cliente' (0 o 1).")
        st.stop()

    X = df.drop("Cliente", axis=1)
    y = df["Cliente"]

    # --- Identificar columnas ---
    num_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

    # --- Preprocesador ---
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    # --- Modelo XGBoost ---
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="aucpr",
        tree_method="hist"
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", xgb)
    ])

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Entrenar ---
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # --- Umbral basado en recall ---
    TARGET_RECALL = 0.60
    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    cand = np.where(rec[:-1] >= TARGET_RECALL)[0]
    i = cand[np.argmax(prec[cand])] if len(cand) else np.argmax(rec[:-1])
    threshold = thr[i]

    # --- Predicciones en todo el dataset ---
    df_resultados = X.copy()
    df_resultados["probabilidad"] = pipe.predict_proba(X)[:, 1]
    df_resultados["prediccion"] = (df_resultados["probabilidad"] >= threshold).astype(int)
    df_resultados["prioridad"] = df_resultados["probabilidad"].apply(
        lambda p: "Alta" if p >= 0.65 else "Media" if p >= threshold else "Baja"
    )

    # Lead Value Score
    if "Presupuesto_Estimado" in df.columns:
        df_resultados["lead_value_score"] = df_resultados["probabilidad"] * df["Presupuesto_Estimado"]
    else:
        st.warning("⚠️ No se encontró 'Presupuesto_Estimado', no se calculará lead_value_score.")
        df_resultados["lead_value_score"] = np.nan

    # --- Visualizaciones ---
    st.subheader("📌 Clientes más importantes (Alta prioridad)")
    st.dataframe(df_resultados[df_resultados["prioridad"] == "Alta"].sort_values("probabilidad", ascending=False).head(20))

    if "lead_value_score" in df_resultados.columns:
        fig1 = px.bar(df_resultados.sort_values("lead_value_score", ascending=False).head(20),
                      x="Cuenta", y="lead_value_score", color="prioridad", title="Top 20 Lead Value Score")
        st.plotly_chart(fig1, use_container_width=True)

    if "Tamaño_Empresa" in df_resultados.columns:
        fig2 = px.histogram(df_resultados, x="Tamaño_Empresa", color="prioridad", barmode="group",
                            title="Prioridad vs Tamaño de Empresa")
        st.plotly_chart(fig2, use_container_width=True)

    if "Sector" in df_resultados.columns:
        fig3 = px.histogram(df_resultados, x="Sector", color="prioridad", barmode="group",
                            title="Prioridad vs Sector")
        st.plotly_chart(fig3, use_container_width=True)

    # Actividad
    for col in ["Emails_Abiertos", "Llamadas_Realizadas", "Visitas_Web"]:
        if col in df_resultados.columns:
            fig = px.box(df_resultados, x="prioridad", y=col, color="prioridad", title=f"{col} por Prioridad")
            st.plotly_chart(fig, use_container_width=True)

    # Distribución general
    fig4 = px.pie(df_resultados, names="prioridad", title="Distribución de Prioridades")
    st.plotly_chart(fig4, use_container_width=True)

    # Sectores con más leads altos
    if "Sector" in df_resultados.columns:
        top_sectores = df_resultados[df_resultados["prioridad"] == "Alta"]["Sector"].value_counts().reset_index()
        top_sectores.columns = ["Sector", "Cantidad"]
        fig5 = px.bar(top_sectores, x="Sector", y="Cantidad", title="Sectores con más Leads de Alta Prioridad")
        st.plotly_chart(fig5, use_container_width=True)

    if "Antiguedad_Cuenta_Meses" in df_resultados.columns:
        fig6 = px.box(df_resultados, x="prioridad", y="Antiguedad_Cuenta_Meses", color="prioridad",
                      title="Prioridad vs Antigüedad de Cuenta (meses)")
        st.plotly_chart(fig6, use_container_width=True)

    st.success("✅ Análisis completado.")
