import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.nb_model = GaussianNB()
        self.lr_model = LogisticRegression(random_state=42)
        
    def load_and_clean_data(self, filepath):
        """Load and clean the diabetes dataset"""
        # Load dataset
        df = pd.read_csv(filepath)
        print("Dataset shape:", df.shape)
        print("\nDataset info:")
        print(df.info())
        
        # Replace zero values with mean for specific columns
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_columns:
            df[col] = df[col].replace(0, df[col].mean())
        
        print("\nMissing values after cleaning:")
        print(df.isnull().sum())
        
        return df
    
    def exploratory_analysis(self, df):
        """Perform EDA on the dataset"""
        # Basic statistics
        print("\nDataset Statistics:")
        print(df.describe())
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.show()
        
        # Feature distributions
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(df.columns):
            axes[i].hist(df[col], bins=20, alpha=0.7)
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        plt.show()
    
    def prepare_data(self, df):
        """Prepare data for training"""
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def train_models(self, X_train, y_train):
        """Train both models"""
        # Train Naive Bayes
        self.nb_model.fit(X_train, y_train)
        
        # Train Logistic Regression
        self.lr_model.fit(X_train, y_train)
        
        print("Models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate and compare models"""
        # Predictions
        nb_pred = self.nb_model.predict(X_test)
        lr_pred = self.lr_model.predict(X_test)
        
        # Accuracies
        nb_accuracy = accuracy_score(y_test, nb_pred)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        
        print(f"\nModel Accuracies:")
        print(f"Naive Bayes: {nb_accuracy:.4f}")
        print(f"Logistic Regression: {lr_accuracy:.4f}")
        
        # Confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        cm_nb = confusion_matrix(y_test, nb_pred)
        cm_lr = confusion_matrix(y_test, lr_pred)
        
        sns.heatmap(cm_nb, annot=True, fmt='d', ax=axes[0], cmap='Blues')
        axes[0].set_title('Naive Bayes Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        sns.heatmap(cm_lr, annot=True, fmt='d', ax=axes[1], cmap='Blues')
        axes[1].set_title('Logistic Regression Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.show()
        
        # Classification reports
        print("\nNaive Bayes Classification Report:")
        print(classification_report(y_test, nb_pred))
        
        print("\nLogistic Regression Classification Report:")
        print(classification_report(y_test, lr_pred))
        
        # ROC Curves
        nb_proba = self.nb_model.predict_proba(X_test)[:, 1]
        lr_proba = self.lr_model.predict_proba(X_test)[:, 1]
        
        fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_proba)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
        
        auc_nb = auc(fpr_nb, tpr_nb)
        auc_lr = auc(fpr_lr, tpr_lr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {auc_nb:.3f})')
        plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curves.png')
        plt.show()
        
        return nb_accuracy, lr_accuracy, auc_nb, auc_lr
    
    def save_models(self):
        """Save trained models and scaler"""
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.nb_model, 'naive_bayes_model.pkl')
        joblib.dump(self.lr_model, 'logistic_regression_model.pkl')
        print("Models saved successfully!")
    
    def predict(self, features, model_type='lr'):
        """Make prediction for new data"""
        features_scaled = self.scaler.transform([features])
        
        if model_type == 'nb':
            prediction = self.nb_model.predict(features_scaled)[0]
            probability = self.nb_model.predict_proba(features_scaled)[0][1]
        else:
            prediction = self.lr_model.predict(features_scaled)[0]
            probability = self.lr_model.predict_proba(features_scaled)[0][1]
        
        return prediction, probability

def main():
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Note: Download diabetes.csv from Kaggle PIMA Indians Diabetes Dataset
    try:
        # Load and clean data
        df = predictor.load_and_clean_data('diabetes.csv')
        
        # Perform EDA
        predictor.exploratory_analysis(df)
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(df)
        
        # Train models
        predictor.train_models(X_train, y_train)
        
        # Evaluate models
        nb_acc, lr_acc, nb_auc, lr_auc = predictor.evaluate_models(X_test, y_test)
        
        # Determine better model
        if lr_acc > nb_acc:
            print(f"\n🏆 Logistic Regression performs better (Accuracy: {lr_acc:.4f}, AUC: {lr_auc:.4f})")
        else:
            print(f"\n🏆 Naive Bayes performs better (Accuracy: {nb_acc:.4f}, AUC: {nb_auc:.4f})")
        
        # Save models
        predictor.save_models()
        
        # Example prediction
        sample_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]  # Sample patient data
        prediction, probability = predictor.predict(sample_data, 'lr')
        
        print(f"\nSample Prediction:")
        print(f"Input: {sample_data}")
        print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
        print(f"Probability: {probability:.4f}")
        
    except FileNotFoundError:
        print("Please download diabetes.csv from Kaggle PIMA Indians Diabetes Dataset")
        print("Place it in the same directory as this script")

if __name__ == "__main__":
    main()