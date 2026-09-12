from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.preprocessing import escalar_datos


def entrenar_modelo_supervisado(X, y):

    # Dividir los datos: 80% entrenamiento y 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Escalar los datos
    X_train_scaled, X_test_scaled = escalar_datos(
        X_train,
        X_test
    )

    # Crear el modelo
    modelo = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Entrenar
    modelo.fit(X_train_scaled, y_train)

    # Realizar predicciones
    y_pred = modelo.predict(X_test_scaled)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    matriz = confusion_matrix(y_test, y_pred)

    print("\n--- MODELO SUPERVISADO ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nMatriz de Confusión:")
    print(matriz)

    return modelo, accuracy, f1, matriz