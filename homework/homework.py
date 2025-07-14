# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import os
import json
import gzip
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)

def load_data(filepath):
    return pd.read_csv(filepath, compression="zip")

def data_clean(df):
    df = df.copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns="ID", inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda val: 4 if val > 4 else val)
    return df

def split_data(train, test):
    X_train = train.drop(columns="default")
    y_train = train["default"]
    X_test = test.drop(columns="default")
    y_test = test["default"]
    return X_train, y_train, X_test, y_test

def create_pipeline():
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    num_cols = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
        "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
        "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("scale", StandardScaler(), num_cols)
    ])

    return Pipeline(steps=[
        ("pre", preprocessor),
        ("pca", PCA()),
        ("select", SelectKBest(score_func=f_classif)),
        ("clf", SVC(kernel="rbf", random_state=42))
    ])

def make_grid_search(pipeline):
    grid = {
        "pca__n_components": [20, 21],
        "select__k": [12],
        "clf__kernel": ["rbf"],
        "clf__gamma": [0.1]
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        verbose=2
    )

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)

def evaluate(estimator, X, y, dataset):
    y_pred = estimator.predict(X)
    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": round(precision_score(y, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "f1_score": round(f1_score(y, y_pred), 4)
    }, y_pred, y

def confusion(y_true, y_pred, dataset):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1])
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1])
        }
    }

def main():
    os.makedirs("files/output", exist_ok=True)

    train_df = data_clean(load_data("files/input/train_data.csv.zip"))
    test_df = data_clean(load_data("files/input/test_data.csv.zip"))
    X_train, y_train, X_test, y_test = split_data(train_df, test_df)

    pipeline = create_pipeline()
    model = make_grid_search(pipeline).fit(X_train, y_train)

    metrics_train, pred_train, _ = evaluate(model, X_train, y_train, "train")
    metrics_test, pred_test, _ = evaluate(model, X_test, y_test, "test")

    cm_train = confusion(y_train, pred_train, "train")
    cm_test = confusion(y_test, pred_test, "test")

    with open("files/output/metrics.json", "w") as f:
        for row in [metrics_train, metrics_test, cm_train, cm_test]:
            f.write(json.dumps(row) + "\n")

    save_model(model, "files/models/model.pkl.gz")

main()
