"""Product Recommendation Model
Reads the merged dataset, trains a shallow Random Forest, prints its scores,
and saves the model. Run from the repo's main folder:
    python scripts/product_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

# 1. load the merged dataset
data = pd.read_csv("data/merged_dataset.csv")

# 2. choose the answer (y) and the input columns (X)
y = data["product_category"]
drop_cols = ["customer_id_legacy", "transaction_id", "customer_id", "purchase_date",
             "product_category", "social_media_platform", "review_sentiment"]
X = data.drop(columns=drop_cols)

# 3. split into training and test rows
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. train the model
model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

# 5. evaluate on the test rows
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("F1-score:", round(f1_score(y_test, pred, average="weighted"), 3))
print("Loss:    ", round(log_loss(y_test, proba, labels=model.classes_), 3))

# 6. save the model and its input columns
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/product_model.pkl")
joblib.dump(list(X.columns), "models/product_model_columns.pkl")
print("Model saved in the models folder.")