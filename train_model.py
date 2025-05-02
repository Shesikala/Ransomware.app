import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier  # Ensure you are using MLPClassifier

# Load dataset
csv_path = r"C:\Users\salom\Desktop\streamlit app\feature_vectors_syscalls_frequency_5_Cat.csv"
df = pd.read_csv(csv_path)

# Remove 'Class' column and split data
X = df.drop(columns=['Class'])
y = df['Class']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP Classifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Save Model and Features
joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "selected_features.pkl")  # Save feature names

print("Model retrained and saved correctly!")
