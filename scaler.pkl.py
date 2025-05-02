import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Path to dataset
csv_path = r"C:\Users\salom\Desktop\streamlit app\feature_vectors_syscalls_frequency_5_Cat.csv"
model_path = r"C:\Users\salom\Desktop\streamlit app\model.pkl"
scaler_path = r"C:\Users\salom\Desktop\streamlit app\scaler.pkl"
features_path = r"C:\Users\salom\Desktop\streamlit app\selected_features.pkl"

# Load the dataset
try:
    df = pd.read_csv(csv_path)
    print("✅ CSV file loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: File not found at {csv_path}. Check the path and try again.")
    exit()
except Exception as e:
    print(f"❌ Error while loading CSV: {e}")
    exit()

# Display available columns
print("🔍 Available Columns in CSV:", df.columns.tolist())

# Remove 'Class' column if it exists (it's the target, not a feature)
if "Class" in df.columns:
    df = df.drop(columns=["Class"])
    print("⚠️ 'Class' column removed from dataset (target variable).")

# Load the trained model to get selected features
model = joblib.load(model_path)
selected_features = model.feature_names_in_  # Get features used in model training

# Ensure dataset includes only selected features
df = df[selected_features]

# Train StandardScaler on selected features
scaler = StandardScaler()
scaler.fit(df)

# Save the trained scaler and selected features
joblib.dump(scaler, scaler_path)
joblib.dump(selected_features, features_path)

print(f"✅ New scaler.pkl saved successfully at {scaler_path}!")
print(f"✅ Selected features saved at {features_path}!")
