
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import joblib
import os

# -----------------------------
# 1. Synthetic Data Generation
# -----------------------------
def generate_explainable_data(n_samples=3000):
    np.random.seed(42)

    age = np.random.randint(18, 65, n_samples)
    income = np.random.randint(25000, 120000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    loan_amount = np.random.randint(5000, 50000, n_samples)

    logits = (
        0.03 * age +
        0.00004 * income +
        0.005 * credit_score -
        0.00005 * loan_amount +
        np.random.normal(0, 1, n_samples)
    )

    target = (logits > np.median(logits)).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "target": target
    })

    return df


# -----------------------------
# 2. Data Loading & Saving
# -----------------------------
def create_and_save_dataset():
    df = generate_explainable_data()
    df.to_csv("explainable_dl_data.csv", index=False)
    print("Synthetic dataset created and saved.")


# -----------------------------
# 3. Data Loading & Inspection
# -----------------------------
def load_and_inspect_data():
    df = pd.read_csv("explainable_dl_data.csv")
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    return df


# -----------------------------
# 4. Preprocessing
# -----------------------------
def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as scaler.pkl")

    return X_scaled, y, X.columns


# -----------------------------
# 5. Deep Learning Model
# -----------------------------
def build_dl_model(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# 6. Training
# -----------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_dl_model(X.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("explainable_dl_model.h5")
    print("Deep learning model saved as explainable_dl_model.h5")

    return model, X_test, y_test


# -----------------------------
# 7. Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test):
    predictions = (model.predict(X_test) > 0.5).astype(int)

    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("Confusion matrix saved as confusion_matrix.png")


# -----------------------------
# 8. Explainability (Permutation Importance)
# -----------------------------
def explain_model(model, X, y, feature_names):
    baseline_preds = (model.predict(X) > 0.5).astype(int)
    baseline_acc = accuracy_score(y, baseline_preds)

    importances = []

    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])

        permuted_preds = (model.predict(X_permuted) > 0.5).astype(int)
        permuted_acc = accuracy_score(y, permuted_preds)

        importances.append(baseline_acc - permuted_acc)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nExplainability - Feature Importance:")
    print(importance_df)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    print("Feature importance plot saved as feature_importance.png")


# -----------------------------
# 9. Inference Function
# -----------------------------
def explainable_predict(input_features):
    scaler = joblib.load("scaler.pkl")
    model = load_model("explainable_dl_model.h5")

    input_features = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_features)

    prediction = model.predict(input_scaled)
    return int(prediction[0][0] > 0.5)


# -----------------------------
# 10. Main Execution
# -----------------------------
def main():
    if not os.path.exists("explainable_dl_data.csv"):
        create_and_save_dataset()

    df = load_and_inspect_data()
    X_scaled, y, feature_names = preprocess_data(df)

    model, X_test, y_test = train_model(X_scaled, y)
    evaluate_model(model, X_test, y_test)

    explain_model(model, X_scaled, y, feature_names)

    sample_input = [35, 60000, 720, 20000]
    prediction = explainable_predict(sample_input)
    print("\nSample Prediction:", prediction)


if __name__ == "__main__":
    main()
