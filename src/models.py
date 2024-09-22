from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_svm(X_train, y_train):
    """
    Train an SVM classifier with an RBF kernel.
    """
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model's accuracy on the test set.
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
