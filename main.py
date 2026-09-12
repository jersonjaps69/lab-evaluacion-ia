from src.data_loader import generar_dataset
from src.supervised_model import entrenar_modelo_supervisado
from src.unsupervised_model import entrenar_modelo_no_supervisado


# 1. Generar dataset
df = generar_dataset()

print("\nPrimeros 5 registros:")
print(df.head())


# 2. Separar características y objetivo
X = df[
    [
        "caracteristica_1",
        "caracteristica_2",
        "caracteristica_3",
        "caracteristica_4"
    ]
]

y = df["objetivo"]


# 3. Ejecutar modelo supervisado
entrenar_modelo_supervisado(X, y)


# 4. Ejecutar modelo no supervisado
entrenar_modelo_no_supervisado(X)