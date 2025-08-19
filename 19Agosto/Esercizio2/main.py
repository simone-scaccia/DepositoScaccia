from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import plot_tree
from sklearn.linear_model import LogisticRegression

# Read AirQualityUCI.csv using ; as the separator
df = pd.read_csv('AirQualityUCI.csv', sep=';', parse_dates=["Date"])

# Print the column names
print(df.columns.tolist())

# Drop empty Date columns
df = df.dropna(subset=["Date", "Time", "PT08.S1(CO)", "PT08.S5(O3)"])

# Convert the column Time (e.g. "10.00.00") to an integer (e.g. 10)
df["Time"] = df["Time"].str.split('.').str[0].astype(int)
df["Week"] = pd.to_datetime(df["Date"], format='%d/%m/%Y').dt.isocalendar().week
df["Month"] = pd.to_datetime(df["Date"], format='%d/%m/%Y').dt.month

# Classify the PT08.S1(CO) column to the median of the column
df["PT08.S1(CO)_class"] = (df["PT08.S1(CO)"] > df["PT08.S1(CO)"].median()).astype(int)
df["PT08.S5(O3)_class"] = (df["PT08.S5(O3)"] > df["PT08.S5(O3)"].median()).astype(int)

# print(' '.join(df["PT08.S1(CO)_class"].astype(str)))

# Select features and target variable
X = df[["Time", "Week", "Month", "PT08.S1(CO)"]]
y = df["PT08.S1(CO)_class"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# -------------------------
# Modello 1 – Decision Tree
# -------------------------
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("Decision Tree:")
print(classification_report(y_test, y_pred_tree, digits=3))

# Visualizzazione albero decisionale (figura dedicata)
plt.figure(figsize=(18, 10))
plot_tree(tree, feature_names=X.columns, class_names=["Buona qualità dell'aria", "Scarsa qualità dell'aria"], filled=True)
plt.title("Albero Decisionale – Dataset Air Quality")
plt.show()

# -------------------------
# Modello 2 – Logistic Regression
# -------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg, digits=3))