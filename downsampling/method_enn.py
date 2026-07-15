from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import GradientBoostingClassifier
from utils import load_and_preprocess_data, evaluate_model

def run_enn():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Applying enn...")
    resampler = EditedNearestNeighbours()\n    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_resampled, y_resampled)
    
    y_pred = clf.predict(X_test)
    evaluate_model(y_test, y_pred, "enn")

if __name__ == "__main__":
    run_enn()
