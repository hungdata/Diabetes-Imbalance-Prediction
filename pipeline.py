import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks, OneSidedSelection
from imblearn.combine import SMOTEENN
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def load_and_preprocess_data(csv_path="diabetes_data.csv"):
    """
    Loads data and applies BMI capping using IQR.
    (Mock implementation assuming 'BMI' and 'Diabetes_binary' exist).
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Dataset {csv_path} not found. Please provide the BRFSS dataset.")
        return None, None, None, None

    # BMI Outlier Capping
    Q1 = df['BMI'].quantile(0.25)
    Q3 = df['BMI'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['BMI'] = df['BMI'].clip(lower=lower_bound, upper=upper_bound)

    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_resampling(X_train, y_train, X_test, y_test):
    """
    Evaluates various resampling techniques on a Gradient Boosting Classifier.
    """
    resamplers = {
        "None": None,
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "ENN": EditedNearestNeighbours(),
        "Tomek Links": TomekLinks(),
        "SMOTE+ENN": SMOTEENN(random_state=42)
    }
    
    results = []
    
    for name, resampler in resamplers.items():
        print(f"Running pipeline for: {name}")
        
        if resampler is not None:
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
            
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X_resampled, y_resampled)
        
        y_pred = clf.predict(X_test)
        
        results.append({
            "Method": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    if X_train is not None:
        df_results = evaluate_resampling(X_train, y_train, X_test, y_test)
        print("\n--- Results Summary ---")
        print(df_results)
