import pandas as pd
from sklearn.datasets import make_classification
import os


def generar_dataset():
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=42
    )

    df = pd.DataFrame(
        X,
        columns=[
            "caracteristica_1",
            "caracteristica_2",
            "caracteristica_3",
            "caracteristica_4"
        ]
    )

    df["objetivo"] = y

    os.makedirs("data", exist_ok=True)

    df.to_csv(
        "data/dataset.csv",
        index=False
    )

    print("Dataset generado correctamente.")

    return df


def cargar_dataset():
    df = pd.read_csv("data/dataset.csv")

    return df