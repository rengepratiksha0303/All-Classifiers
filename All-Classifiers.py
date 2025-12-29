import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# -------------------------
# Initial Setup
# -------------------------
sns.set(style="whitegrid")
np.random.seed(42)

# -------------------------
# Load dataset
# -------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print("Dataset loaded.")
print(f"Features ({len(feature_names)}): {list(feature_names)}")
print(f"Target classes: {list(target_names)}")

# -------------------------
# Explore class distribution
# -------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="pastel")
plt.xticks([0, 1], target_names)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# -------------------------
# Split the dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(y_train)}, Testing samples: {len(y_test)}")

# -------------------------
# Define models and hyperparameter grids
# -------------------------
print("\nSetting up models and parameters...")
dt = DecisionTreeClassifier(random_state=42)
dt_params = {'max_depth': [3, 5], 'min_samples_split': [2, 5]}

rf = RandomForestClassifier(random_state=42)
rf_params = {'n_estimators': [50], 'max_depth': [5], 'min_samples_split': [2]}

ab = AdaBoostClassifier(random_state=42)
ab_params = {'n_estimators': [50], 'learning_rate': [0.1]}

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_params = {'n_estimators': [50], 'max_depth': [3], 'learning_rate': [0.1]}

cat = CatBoostClassifier(verbose=0, random_state=42)
cat_params = {'iterations': [100], 'depth': [3], 'learning_rate': [0.1]}

models = {
    "Decision Tree": (dt, dt_params),
    "Random Forest": (rf, rf_params),
    "AdaBoost": (ab, ab_params),
    "XGBoost": (xgb, xgb_params),
    "CatBoost": (cat, cat_params)
}

# -------------------------
# Training and Hyperparameter Tuning
# -------------------------
best_models = {}
print("\nStarting training and hyperparameter tuning...")
for name, (clf, params) in models.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(clf, params, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    print(f"Best parameters for {name}: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

# -------------------------
# Evaluate models on test set
# -------------------------
results = {}
print("\nEvaluating models on the test set...")
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=target_names))

# -------------------------
# Visualize Accuracy Comparison
# -------------------------
plt.figure(figsize=(10,6))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="coolwarm")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.ylim(0.8, 1)
plt.show()

# -------------------------
# Plot Confusion Matrices
# -------------------------
print("\nConfusion Matrices:")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (name, model) in enumerate(best_models.items()):
    if idx >= 6:
        break
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx])
    axes[idx].set_title(f"{name}")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------
# ROC Curves for All Models
# -------------------------
print("\nROC Curves:")
plt.figure(figsize=(10,8))
for name, model in best_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# -------------------------
# Feature Importances (for tree-based models)
# -------------------------
print("\nFeature Importances:")
for name, model in best_models.items():
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,5))
        sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
        plt.title(f"{name} Feature Importances")
        plt.show()

# -------------------------
# Learning Curves (Optional, for Decision Tree and Random Forest)
# -------------------------
print("\nPlotting Learning Curves:")
for name in ["Decision Tree", "Random Forest"]:
    model = best_models[name]
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, label="Training Score", marker='o')
    plt.plot(train_sizes, test_mean, label="Cross-validation Score", marker='o')
    plt.title(f"Learning Curve for {name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# -------------------------
# Analyze Prediction Errors (Optional)
# -------------------------
print("\nDistribution of Prediction Errors:")
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    errors = y_test - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(errors, bins=3, kde=True, color="red")
    plt.title(f"Prediction Errors for {name}")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()

# -------------------------
# Save the Best Model using pickle
# -------------------------
best_model_name = max(results, key=results.get)
best_model = best_models[best_model_name]
print(f"\nBest model is {best_model_name} with accuracy {results[best_model_name]:.4f}")

filename = 'best_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)
print(f"Saved best model to {filename}")

# -------------------------
# Load and Validate the Model
# -------------------------
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

final_accuracy = accuracy_score(y_test, loaded_model.predict(X_test))
print(f"Loaded model accuracy: {final_accuracy:.4f}")
