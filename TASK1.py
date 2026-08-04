import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("StudentPerformanceFactors.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

df.drop_duplicates(inplace=True)

df["Teacher_Quality"].fillna(df["Teacher_Quality"].mode()[0], inplace=True)
df["Parental_Education_Level"].fillna(df["Parental_Education_Level"].mode()[0], inplace=True)
df["Distance_from_Home"].fillna(df["Distance_from_Home"].mode()[0], inplace=True)

df = pd.get_dummies(df, drop_first=True, dtype=int)

plt.figure(figsize=(8,5))
plt.scatter(df["Hours_Studied"], df["Exam_Score"])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Exam_Score"], bins=15)
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Exam Scores")
plt.show()

X = df[["Hours_Studied"]]
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLinear Regression")
print("MAE =", mean_absolute_error(y_test, y_pred))
print("MSE =", mean_squared_error(y_test, y_pred))
print("R2 =", r2_score(y_test, y_pred))

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(X_test["Hours_Studied"], y_test)
plt.plot(X_test["Hours_Studied"], y_pred, color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression")
plt.show()

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(df[["Hours_Studied"]])

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly,
    y,
    test_size=0.2,
    random_state=42
)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train_poly)

poly_pred = poly_model.predict(X_test_poly)

print("\nPolynomial Regression")
print("MAE =", mean_absolute_error(y_test_poly, poly_pred))
print("MSE =", mean_squared_error(y_test_poly, poly_pred))
print("R2 =", r2_score(y_test_poly, poly_pred))

plt.figure(figsize=(8,5))
plt.scatter(y_test_poly, poly_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Polynomial Regression")
plt.show()

feature_sets = [
    ["Hours_Studied"],
    ["Hours_Studied", "Sleep_Hours"],
    ["Hours_Studied", "Attendance", "Previous_Scores"],
    [col for col in df.columns if col != "Exam_Score"]
]

print("\nFeature Comparison")

for features in feature_sets:

    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("--------------------------------------")
    print("Features:", features)
    print("MAE =", mean_absolute_error(y_test, pred))
    print("MSE =", mean_squared_error(y_test, pred))
    print("R2 =", r2_score(y_test, pred))