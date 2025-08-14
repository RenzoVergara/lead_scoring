# lead_scoring.py
import pandas as pd
import numpy as np
import random
import joblib
import json
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report
)
from xgboost import XGBClassifier

# =========================
# 1. Generar Dataset Sintético
# =========================
def generar_dataset_sintetico(total_filas=33334, total_clientes=1000, seed=42):
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)

    total_no_clientes = total_filas - total_clientes

    def generar_fila(cliente):
        cuenta = fake.company()
        contacto = fake.name()
        celular = fake.msisdn()[:9]
        correo = fake.email()
        stage = np.random.choice(["qualifying", "selling", "negotiating", "closing"], p=[0.25]*4)
        puesto = np.random.choice([
            "CEO", "CTO", "Gerente de Marketing", "Ejecutivo de Ventas",
            "Analista", "Consultor", "Especialista de Producto"
        ])

        visitas_web = max(0, np.random.poisson(4) + (np.random.randint(0, 3) if cliente else np.random.randint(-2, 2)))
        emails_abiertos = max(0, np.random.poisson(2) + (np.random.randint(0, 2) if cliente else np.random.randint(-1, 2)))
        llamadas_realizadas = max(0, np.random.poisson(1) + (np.random.randint(0, 2) if cliente else np.random.randint(-1, 1)))
        presupuesto_estimado = np.random.randint(500, 50000) + (np.random.randint(0, 5000) if cliente else np.random.randint(-3000, 3000))
        sector = np.random.choice(["Tecnología", "Retail", "Manufactura", "Servicios", "Educación", "Salud"])
        tamaño_empresa = np.random.choice(["Pequeña", "Mediana", "Grande"], p=[0.5, 0.3, 0.2])
        antiguedad_cuenta_meses = np.random.randint(1, 48)
        interacciones_totales = visitas_web + emails_abiertos + llamadas_realizadas

        return {
            "Cuenta": cuenta,
            "Contacto": contacto,
            "Puesto": puesto,
            "Celular": celular,
            "Correo_electronico": correo,
            "Stage": stage,
            "Visitas_Web": visitas_web,
            "Emails_Abiertos": emails_abiertos,
            "Llamadas_Realizadas": llamadas_realizadas,
            "Presupuesto_Estimado": presupuesto_estimado,
            "Sector": sector,
            "Tamaño_Empresa": tamaño_empresa,
            "Antiguedad_Cuenta_Meses": antiguedad_cuenta_meses,
            "Interacciones_Totales": interacciones_totales,
            "Cliente": cliente
        }

    datos_clientes = [generar_fila(1) for _ in range(total_clientes)]
    datos_no_clientes = [generar_fila(0) for _ in range(total_no_clientes)]
    datos = datos_clientes + datos_no_clientes
    random.shuffle(datos)

    df = pd.DataFrame(datos)
    return df

# =========================
# 2. Preprocesamiento
# =========================
def limpiar_datos(df):
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Desconocido")
    return df

# =========================
# 3. Entrenamiento XGBoost
# =========================
def entrenar_xgboost(X_train, y_train, X_test, y_test, best_params, target_recall=0.60):
    num_features = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )

    xgb = XGBClassifier(**best_params)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", xgb)
    ])

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    cand = np.where(rec[:-1] >= target_recall)[0]
    if len(cand):
        i = cand[np.argmax(prec[cand])]
    else:
        i = np.argmax(rec[:-1])
    thr_recall = float(thr[i])

    print(f"Umbral(Recall≥{target_recall:.0%}): {thr_recall:.4f} | Precision: {prec[i]:.4f} | Recall: {rec[i]:.4f}")
    y_pred = (y_proba >= thr_recall).astype(int)
    print("\nReporte:\n", classification_report(y_test, y_pred, digits=4))

    return pipe, thr_recall

# =========================
# 4. Guardar Modelo
# =========================
def guardar_modelo(pipe, threshold, target_recall, path_model="xgb_lead_scoring_pipeline.joblib", path_meta="xgb_lead_scoring_meta.json"):
    joblib.dump(pipe, path_model)
    json.dump({"threshold_recall_first": threshold, "recall_target": target_recall}, open(path_meta, "w"))
    print(f"✅ Modelo guardado en {path_model} y meta en {path_meta}")

# =========================
# 5. Generar Ranking
# =========================
def generar_ranking(X, pipe, threshold, path_excel="ranking_leads.xlsx", path_csv="ranking_leads.csv"):
    df_resultados = X.copy().reset_index(drop=True)
    df_resultados["probabilidad"] = pipe.predict_proba(X)[:, 1]
    df_resultados["prediccion"] = (df_resultados["probabilidad"] >= threshold).astype(int)

    def clasificar_prioridad(prob):
        if prob >= 0.70:
            return "Alta"
        elif prob >= threshold:
            return "Media"
        else:
            return "Baja"

    df_resultados["prioridad"] = df_resultados["probabilidad"].apply(clasificar_prioridad)
    df_resultados = df_resultados.sort_values("probabilidad", ascending=False).reset_index(drop=True)
    df_resultados.to_excel(path_excel, index=False)
    df_resultados.to_csv(path_csv, index=False)
    print(f"✅ Ranking guardado en {path_excel} y {path_csv}")
    return df_resultados

# =========================
# 6. Main (Ejecución Completa)
# =========================
if __name__ == "__main__":
    # Generar dataset
    df = generar_dataset_sintetico()
    df = limpiar_datos(df)

    # Separar variables
    X = df.drop("Cliente", axis=1)
    y = df["Cliente"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Hiperparámetros óptimos (de Optuna)
    best_params = {
        'n_estimators': 463,
        'learning_rate': 0.01367766664555449,
        'max_depth': 4,
        'min_child_weight': 4.130045906042306,
        'subsample': 0.7770797743906411,
        'colsample_bytree': 0.9691613610509624,
        'gamma': 2.8011024901326653,
        'reg_lambda': 3.7376682216067323,
        'reg_alpha': 0.007676740816505292,
        'scale_pos_weight': 28.64684818682142,
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'max_delta_step': 1,
        'n_jobs': -1,
        'random_state': 42
    }

    # Entrenar modelo
    pipe, threshold = entrenar_xgboost(X_train, y_train, X_test, y_test, best_params, target_recall=0.60)

    # Guardar modelo
    guardar_modelo(pipe, threshold, 0.60)

    # Generar ranking
    generar_ranking(X_test, pipe, threshold)
