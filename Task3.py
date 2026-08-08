import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

df = pd.read_csv("covtype.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"] - 1

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest")
print("Accuracy =", rf_accuracy)

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=[
            "Cover Type 1",
            "Cover Type 2",
            "Cover Type 3",
            "Cover Type 4",
            "Cover Type 5",
            "Cover Type 6",
            "Cover Type 7"
        ]
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_pred,
    display_labels=[1, 2, 3, 4, 5, 6, 7]
)

plt.title("Random Forest Confusion Matrix")
plt.show()

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features")
print(importance.head(10))

plt.figure(figsize=(10, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=7,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\nXGBoost")
print("Accuracy =", xgb_accuracy)

print(
    classification_report(
        y_test,
        xgb_pred,
        target_names=[
            "Cover Type 1",
            "Cover Type 2",
            "Cover Type 3",
            "Cover Type 4",
            "Cover Type 5",
            "Cover Type 6",
            "Cover Type 7"
        ]
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    xgb_pred,
    display_labels=[1, 2, 3, 4, 5, 6, 7]
)

plt.title("XGBoost Confusion Matrix")
plt.show()

xgb_importance = pd.Series(
    xgb.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 XGBoost Important Features")
print(xgb_importance.head(10))

plt.figure(figsize=(10, 6))
xgb_importance.head(10).sort_values().plot(kind="barh")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

print("\nModel Comparison")
print("Random Forest Accuracy =", rf_accuracy)
print("XGBoost Accuracy =", xgb_accuracy)