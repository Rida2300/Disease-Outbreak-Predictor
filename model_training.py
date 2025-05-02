import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the data
data = pd.read_csv('data.csv')

# Assuming the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'disease_model.pkl')

print("Model training complete. Model saved as disease_model.pkl.")
