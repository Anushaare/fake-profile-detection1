# Import libraries
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Step 1: Sample dataset
# Features: [experience, education, skills, projects, certifications]
X = np.array([
    [2, 0, 3, 1, 0],
    [5, 1, 6, 3, 2],
    [1, 0, 2, 0, 0],
    [7, 2, 10, 5, 3],
    [3, 1, 4, 2, 1],
    [6, 2, 8, 4, 2],
    [1, 0, 1, 0, 0],
    [4, 1, 5, 2, 1]
])

# Labels: 0 = Genuine, 1 = Fake
y = np.array([1, 0, 1, 0, 0, 0, 1, 0])

# Step 2: Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Step 4: Save model
pickle.dump(model, open("model.pkl", "wb"))

# Step 5: Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ model.pkl and scaler.pkl created successfully!")