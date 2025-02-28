# Breast Cancer Detection using Machine Learning

## Project Overview
This project aims to detect breast cancer using machine learning techniques. It utilizes the **Breast Cancer Wisconsin Dataset** to classify tumors as **Malignant (M)** or **Benign (B)** based on various cell features.

## Dataset
- **Source**: UCI Machine Learning Repository
- **Features**: 30 numerical attributes representing cell characteristics
- **Target Variable**: `diagnosis` (M = Malignant, B = Benign)

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Data Preprocessing
1. **Handling Missing Values**: Dropped unnecessary `Unnamed: 32` column.
2. **Encoding Target Variable**: Converted `M` to `1` (Malignant) and `B` to `0` (Benign).
3. **Feature Scaling**: Used `StandardScaler()` for normalization.
4. **Splitting Data**: 80% training, 20% testing.

## Exploratory Data Analysis (EDA)
- Visualized class distribution using a **countplot**.
- Computed a **correlation matrix** to understand feature relationships.

## Model Training & Evaluation
- **Algorithms Used**: Logistic Regression, Random Forest, SVM, etc.
- **Evaluation Metrics**:
  - Accuracy
  - Precision & Recall
  - ROC-AUC Curve

## How to Run the Project
```python
# Load dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
breast = pd.read_csv("breast_cancer_data.csv")

# Preprocessing
breast.drop(["Unnamed: 32"], axis=1, inplace=True)
breast['diagnosis'] = breast['diagnosis'].map({'M': 1, 'B': 0})
X = breast.drop("diagnosis", axis=1)
y = breast["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

## Future Improvements
- Feature selection to reduce dimensionality.
- Hyperparameter tuning using `GridSearchCV`.
- Deploying as a web application.

## License
This project is open-source under the MIT License.

---

