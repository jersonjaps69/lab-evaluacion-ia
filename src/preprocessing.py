from sklearn.preprocessing import StandardScaler


def escalar_datos(X_train, X_test):
    scaler = StandardScaler()

    # El scaler se ajusta SOLO con los datos de entrenamiento
    X_train_scaled = scaler.fit_transform(X_train)

    # Los datos de prueba solo se transforman
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled