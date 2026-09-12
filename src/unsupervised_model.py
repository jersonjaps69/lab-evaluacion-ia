from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def entrenar_modelo_no_supervisado(X):

    # 1. Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Reducir de 4 características a 2 componentes con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 3. Calcular varianza explicada acumulada
    varianza_acumulada = pca.explained_variance_ratio_.sum()

    # 4. Aplicar K-Means con k = 3
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_pca)

    # 5. Calcular Silhouette Score
    silhouette = silhouette_score(X_pca, clusters)

    print("\n--- MODELO NO SUPERVISADO ---")

    print(
        f"Varianza explicada acumulada: "
        f"{varianza_acumulada:.4f}"
    )

    print(
        f"Silhouette Score: "
        f"{silhouette:.4f}"
    )

    # 6. Visualización 2D
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=clusters
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=200,
        label="Centroides"
    )

    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title("Clustering K-Means sobre componentes PCA")

    plt.legend()
    plt.grid()

    plt.savefig(
        "pca_kmeans.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return X_pca, clusters, silhouette, varianza_acumulada