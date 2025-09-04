import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import optuna

# ================================
# Load Data
# ================================
train = pd.read_csv("./titanic (1)/train.csv")
test = pd.read_csv("./titanic (1)/test.csv")

X_train = train.drop(["Survived", "Name", "Cabin"], axis=1)
y_train = train["Survived"]
X_test = test.copy()

# ================================
# Handle Missing Values
# ================================
X_train["Age"] = X_train["Age"].fillna(X_train["Age"].median())
X_train["Embarked"] = X_train["Embarked"].fillna(X_train["Embarked"].mode()[0])

X_test["Age"] = X_test["Age"].fillna(X_test["Age"].median())
X_test["Fare"] = X_test["Fare"].fillna(X_test["Fare"].median())
X_test["Embarked"] = X_test["Embarked"].fillna(X_test["Embarked"].mode()[0])

# ================================
# Encode Categorical Features
# ================================
le = LabelEncoder()
X_train["Sex_encoded"] = le.fit_transform(X_train["Sex"])
X_train["Embarked_encoded"] = le.fit_transform(X_train["Embarked"])
X_train.drop(["Sex", "Embarked", "Ticket"], axis=1, inplace=True)

X_test["Sex_encoded"] = le.fit_transform(X_test["Sex"])
X_test["Embarked_encoded"] = le.fit_transform(X_test["Embarked"])
passenger_ids = X_test["PassengerId"]
X_test.drop(["Sex", "Embarked", "Ticket", "Name", "Cabin", "PassengerId"], axis=1, inplace=True)

# ================================
# Hyperparameter Optimization with Optuna
# ================================
def objective(trial):
    rf = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        max_depth=trial.suggest_int("max_depth", 2, 25),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        random_state=42,
        n_jobs=-1,
    )
    return cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best Parameters:", study.best_params)
print("Best CV Score:", study.best_value)

# ================================
# Train Final Model
# ================================
best_params = study.best_params
model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
print("Cross-validated Accuracy:", cv_score)

# ================================
# Generate Submission
# ================================
y_pred = model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": y_pred
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

