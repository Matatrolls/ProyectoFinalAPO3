from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def create_random_forest(n_estimators=100, max_depth=None, random_state=42):
    """
    Retorna un clasificador Random Forest configurado para clasificación.
    Utiliza n_jobs=-1 para paralelizar el entrenamiento en todos los núcleos.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

def create_svm(C=1.0, kernel='rbf', gamma='scale', random_state=42):
    """
    Retorna un clasificador Support Vector Machine (SVM) con kernel RBF.
    probability=True es necesario para poder obtener probabilidades de clase (predict_proba)
    durante la inferencia y para evaluar umbrales de confianza (OOD).
    """
    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        random_state=random_state
    )
