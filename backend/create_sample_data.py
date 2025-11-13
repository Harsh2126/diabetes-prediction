import pandas as pd
import numpy as np

# Create sample diabetes dataset for testing
np.random.seed(42)

# Generate sample data similar to PIMA Indians Diabetes Dataset
n_samples = 768

data = {
    'Pregnancies': np.random.randint(0, 17, n_samples),
    'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
    'BloodPressure': np.random.normal(70, 15, n_samples).clip(0, 122),
    'SkinThickness': np.random.normal(20, 10, n_samples).clip(0, 99),
    'Insulin': np.random.normal(80, 100, n_samples).clip(0, 846),
    'BMI': np.random.normal(32, 7, n_samples).clip(0, 67),
    'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
    'Age': np.random.randint(21, 81, n_samples)
}

# Create outcome based on features (simplified logic)
df = pd.DataFrame(data)
outcome_prob = (
    (df['Glucose'] > 140) * 0.3 +
    (df['BMI'] > 30) * 0.2 +
    (df['Age'] > 50) * 0.2 +
    (df['Pregnancies'] > 5) * 0.1 +
    (df['BloodPressure'] > 80) * 0.1 +
    np.random.uniform(0, 0.1, n_samples)
)

df['Outcome'] = (outcome_prob > 0.5).astype(int)

# Add some zero values to simulate real dataset
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    zero_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[zero_indices, col] = 0

# Save to CSV
df.to_csv('diabetes.csv', index=False)
print(f"Sample dataset created with {len(df)} records")
print(f"Diabetes cases: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
print("\nDataset saved as 'diabetes.csv'")
print("\nFirst 5 rows:")
print(df.head())