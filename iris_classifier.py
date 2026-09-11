import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

print("=== Iris Dataset Loaded ===")
print(f"Total Samples: {len(df)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nSpecies Count:")
print(df['species_name'].value_counts())

# Train the model
X = df[iris.feature_names]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Model Performance ===")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
print("=== Feature Importance ===")
importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)

# Chart
plt.figure(figsize=(10, 5))
plt.bar(importance['feature'], importance['importance'], color='lightgreen')
plt.title('Feature Importance - Iris Classifier')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
# Predict
print("\n=== Iris Flower Predictor ===")
print("Enter flower measurements to predict the species!\n")

while True:
    inp = input("Enter measurements (or 'quit' to exit): sepal_length, sepal_width, petal_length, petal_width\n")
    if inp.lower() == 'quit':
        break
    try:
        values = [float(x) for x in inp.split(',')]
        prediction = model.predict([values])[0]
        species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        print(f"\nPredicted Species: {species[prediction]} 🌸\n")
    except:
        print("Invalid input! Enter 4 numbers separated by commas.")