# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load preprocessed data
df = pd.read_csv('data/processed_heart_data.csv')

# Features and Target
X = df.drop('HeartDisease', axis=1)

# Make sure 'DatasetSource' is NOT present anymore
if 'DatasetSource' in X.columns:
    X = X.drop('DatasetSource', axis=1)

y = df['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train Accuracy: {train_score:.2f}")
print(f"Test Accuracy: {test_score:.2f}")

# Save model
with open('models/heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved at 'models/heart_disease_model.pkl'")
