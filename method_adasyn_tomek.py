from imblearn.over_sampling import ADASYN\nfrom imblearn.under_sampling import TomekLinks
from sklearn.ensemble import GradientBoostingClassifier
from utils import load_and_preprocess_data, evaluate_model

def run_adasyn_tomek():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Applying adasyn_tomek...")
    ada = ADASYN(random_state=42)\n    X_tmp, y_tmp = ada.fit_resample(X_train, y_train)\n    tl = TomekLinks()\n    X_resampled, y_resampled = tl.fit_resample(X_tmp, y_tmp)
    
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_resampled, y_resampled)
    
    y_pred = clf.predict(X_test)
    evaluate_model(y_test, y_pred, "adasyn_tomek")

if __name__ == "__main__":
    run_adasyn_tomek()
