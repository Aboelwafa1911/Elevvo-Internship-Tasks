import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE

df = pd.read_csv("loan_approval_dataset.csv")

df.columns = df.columns.str.strip()

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

categorical_columns = ["education", "self_employed"]

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

y = y.astype(str).str.strip()

if y.str.lower().isin(["approved", "rejected"]).all():
    y = y.str.lower().map({
        "rejected": 0,
        "approved": 1
    })
else:
    y = pd.to_numeric(y)

print("\nClass Distribution")
print(y.value_counts())

plt.figure(figsize=(6, 4))
y.value_counts().plot(kind="bar")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applications")
plt.title("Loan Approval Distribution")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

logistic_pred = logistic_model.predict(X_test_scaled)

print("\nLogistic Regression")
print("Accuracy =", accuracy_score(y_test, logistic_pred))
print(classification_report(y_test, logistic_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    logistic_pred
)

plt.title("Logistic Regression Confusion Matrix")
plt.show()

tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

print("\nDecision Tree")
print("Accuracy =", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    tree_pred
)

plt.title("Decision Tree Confusion Matrix")
plt.show()

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nClass Distribution After SMOTE")
print(y_train_smote.value_counts())

logistic_smote = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_smote.fit(
    scaler.fit_transform(X_train_smote),
    y_train_smote
)

logistic_smote_pred = logistic_smote.predict(
    scaler.transform(X_test)
)

print("\nLogistic Regression with SMOTE")
print("Accuracy =", accuracy_score(y_test, logistic_smote_pred))
print(classification_report(y_test, logistic_smote_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    logistic_smote_pred
)

plt.title("Logistic Regression with SMOTE")
plt.show()

tree_smote = DecisionTreeClassifier(
    random_state=42,
    max_depth=10
)

tree_smote.fit(X_train_smote, y_train_smote)

tree_smote_pred = tree_smote.predict(X_test)

print("\nDecision Tree with SMOTE")
print("Accuracy =", accuracy_score(y_test, tree_smote_pred))
print(classification_report(y_test, tree_smote_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    tree_smote_pred
)

plt.title("Decision Tree with SMOTE")
plt.show()

print("\nModel Comparison")

print(
    "Logistic Regression F1 =",
    classification_report(
        y_test,
        logistic_pred,
        output_dict=True
    )["weighted avg"]["f1-score"]
)

print(
    "Decision Tree F1 =",
    classification_report(
        y_test,
        tree_pred,
        output_dict=True
    )["weighted avg"]["f1-score"]
)

print(
    "Logistic Regression with SMOTE F1 =",
    classification_report(
        y_test,
        logistic_smote_pred,
        output_dict=True
    )["weighted avg"]["f1-score"]
)

print(
    "Decision Tree with SMOTE F1 =",
    classification_report(
        y_test,
        tree_smote_pred,
        output_dict=True
    )["weighted avg"]["f1-score"]
)