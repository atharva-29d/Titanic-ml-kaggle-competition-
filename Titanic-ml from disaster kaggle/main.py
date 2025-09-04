import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.metrics import classification_report, confusion_matrix
import warnings

df = pd.read_csv('./titanic (1)/train.csv')

# print(df.head())
# print(df.describe(include = "all"))

x_train = df.drop(['Survived', 'Name', 'Cabin'], axis = 1)
y_train = df['Survived']

x_test = pd.read_csv('./titanic (1)/test.csv')

# print(x_train.columns)
# print(x_train.isnull().sum())

for column in x_train.columns:
    if x_train[column].dtype == "object":
        print(column.upper(), ':', x_train[column].nunique())
        print(x_train[column].value_counts().sort_values())
        print("\n")


x_train['Age'].fillna(x_train['Age'].median(), inplace = True)

x_train['Embarked'].fillna(x_train['Embarked'].mode()[0], inplace = True)

print(x_train.isnull().sum())
#
# cat_cols = [col for col in x_train.columns if x_train[col].dtype == 'object']
# print(cat_cols)

##ENCODING USING PANDAS
# encoded_df = pd.get_dummies(x_train , columns =['Sex', 'Embarked'])
# print(encoded_df)

##ONE HOT ENCODING
# encoder = OneHotEncoder(sparse_output = False)
# one_hot_encoded  = encoder.fit_transform(x_train[cat_cols])
# # print(one_hot_encoded)
#
# one_hot_df = pd.DataFrame(one_hot_encoded, columns = encoder.get_feature_names_out(cat_cols))
# # print(one_hot_df)
#
# x_train = pd.concat([x_train.drop(cat_cols, axis = 1), one_hot_df], axis = 1)


#LABEL ENCODER
le = LabelEncoder()
x_train['SEX_encoded'] = le.fit_transform(x_train['Sex'])
x_train['Embarked_encoded'] = le.fit_transform(x_train['Embarked'])
print(x_train)

x_train.drop(['Sex', 'Embarked', 'Ticket'], axis = 1, inplace = True)
print(x_train)

##TESTING DATA CHANGES
##TESTING DATA CHANGES
x_test['Age'].fillna(x_test['Age'].median(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].median(), inplace=True)
x_test['Embarked'].fillna(x_test['Embarked'].mode()[0], inplace=True)

# Use separate encoders for test
le_sex = LabelEncoder()
le_sex.fit(x_test['Sex'])   # fit on train column
x_test['SEX_encoded'] = le_sex.transform(x_test['Sex'])

le_embarked = LabelEncoder()
le_embarked.fit(x_test['Embarked'])   # fit on train column
x_test['Embarked_encoded'] = le_embarked.transform(x_test['Embarked'])

# Drop categorical + unused cols
x_test.drop(['Sex', 'Embarked', 'Ticket', 'Name', 'Cabin'], axis=1, inplace=True)

# Save PassengerId
passenger_ids = x_test['PassengerId']
x_test.drop(['PassengerId'], axis=1, inplace=True)



# numeric_cols = x_train.select_dtypes(include = ['int64', 'float64']).columns
#
# plt.figure(figsize=(12,6))
# sns.boxplot(x_train[numeric_cols])
# plt.xticks(rotation = 45)
# plt.title("boxplots for all numeric feature")
# plt.show()

# train_with_target = x_train.copy()
# train_with_target['Survived'] = y_train

# corr = train_with_target.corr()
# plt.figure(figsize=(15,12))
# sns.heatmap(corr[['Survived']].sort_values(by='Survived', ascending = False), annot = True, cmap = 'coolwarm')
# plt.title("Feature selection")
# plt.show()
#
#
# train_with_target = x_train.copy()
# train_with_target['Survived'] = y_train
# corr = train_with_target.corr()
# plt.figure(figsize=(10,6))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Heatmap")
# plt.show()

x_train.drop(["PassengerId"], inplace = True, axis = 1)

#coming to the actual

# rf = RandomForestClassifier()
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, None],
#     'min_samples_leaf': [1, 2],
#     'min_samples_split': [2, 5],
#     'max_features': ['sqrt']
# }
#
# random_search = RandomizedSearchCV(estimator=rf,
#                                    param_distributions=param_grid,
#                                    n_iter=20,  # number of random combos
#                                    cv=5,
#                                    scoring='accuracy',
#                                    n_jobs=-1,
#                                    random_state=42,
#                                    verbose=2)
# random_search.fit(x_train, y_train)
#
# print("Best Parameters:", random_search.best_params_)
# print("Best Cross-Validation Score:", random_search.best_score_)


def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 25)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['log2'])

    # Define model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation score
    score = cross_val_score(rf, x_train, y_train, cv=5, scoring='accuracy').mean()
    return score

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # number of trials

# print("Best parameters:", study.best_params)
# print("Best CV score:", study.best_value)

best_params = study.best_params
model = RandomForestClassifier(**best_params, random_state = 42 , n_jobs = -1)
model.fit(x_train, y_train)

cv_score = cross_val_score(model , x_train , y_train, cv = 5, scoring = "accuracy").mean()
print("Cross-validated Accuracy:", cv_score)

y_pred = model.predict(x_test)
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": y_pred
})
submission.to_csv("submission.csv", index=False)

y_test = pd.read_csv("submission.csv")
y_test = y_test['Survived']
cv_score_test = cross_val_score(model , x_test , y_test, cv = 5, scoring = "accuracy").mean()
print("Cross-validated Accuracy:", cv_score_test)