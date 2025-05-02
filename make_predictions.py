import pandas as pd
import joblib

# Load the trained model
model = joblib.load('disease_model.pkl')
# Example: New data (replace these numbers with real input values)
# Make sure the features match the training data format
new_data = pd.DataFrame({
    'feature1': [1],
    'feature2': [2],
    'feature3': [3],
    'feature4': [4]
})
# Make prediction
prediction = model.predict(new_data)
print(f"Predicted class: {prediction[0]}")
