# linkscanner
ONNX Model and feature scaling logic for the LinkScan Android app.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --- 1. Separate Features and Target ---
# 'id' is a serial number and should not be used as a feature.
X = df_pruned.drop(['Label', 'id'], axis=1) 
y = df_pruned['Label']              

# 2. Split the data before scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, # 20% for testing
    random_state=42, 
    stratify=y 
)

# --- 3. Feature Scaling (Standardization) ---
# Fit the scaler only on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Convert back to DataFrame for Feature Importance step
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)


# --- 4. Train Random Forest and Extract Importance ---
print("\n--- Training Random Forest for Feature Importance Ranking ---")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled_df, y_train)

# 5. Extract feature importances
importances = rf_model.feature_importances_
feature_names = X_train_scaled_df.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# --- 6. Visualize and Select the Top 10 Features ---
plt.figure(figsize=(12, 10))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Random Forest Feature Importance (Gini)")
plt.title("Feature Importance Ranking for Minimal Mobile Model")
plt.gca().invert_yaxis() 
plt.show()


# Select the Top N features (The point where the importance dramatically drops)
N = 10 
top_n_features = feature_importance_df['Feature'].head(N).tolist()

print("\n--- Top 10 Selected Lexical Features for Mobile App ---")
print(top_n_features)
