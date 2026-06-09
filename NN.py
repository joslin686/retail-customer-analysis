# ── Import libraries ───────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Upload the CSV file from your computer
from google.colab import files
uploaded = files.upload()

df = pd.read_csv("customer_features2.csv")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ── Step 2: Selecting the same 5 features used in the Decision Tree ──────────────


X = df[['ProductDiversity', 'AvgBasketValue', 'DaysSinceFirstPurchase',
        'AvgQuantity', 'AvgUnitPrice']]
y = df['IsChurned']

print("\nFeatures shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts())

# ── Step 3: Encode IsChurned from Yes/No to 1/0 ───────────────────────────────


# Yes (churned) = 1, No (not churned) = 0
y = (y == 'Yes').astype(int)
print("\nEncoded class distribution:")
print(y.value_counts())

# ── Step 4: Split into train/test sets (70/30) ────────────────────────────────


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nTraining set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# ── Step 5: Scaling the features ────────────────────────────────────────────────


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("\nScaling done!")

# ── Step 6: Build the Neural Network ──────────────────────────────────────────

# We use 5 neurons in input_shape because we now have 5 features

model = Sequential([

    # First hidden layer — 64 neurons
  
    Dense(64, activation='relu', input_shape=(5,)),
    Dropout(0.3),

    # Second hidden layer — 32 neurons
    Dense(32, activation='relu'),
    Dropout(0.3),

    # Third hidden layer — 16 neurons
    
    Dense(16, activation='relu'),
    Dropout(0.2),

    # Output layer — 1 neuron, sigmoid for Yes/No probability
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
# ── Step 7: Train the model ───────────────────────────────────────────────────


class_weights = {0: 1, 1: 2}

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)
# ── Step 8: Plot training history ─────────────────────────────────────────────


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# ── Step 9: Evaluate the model ────────────────────────────────────────────────

# Convert probabilities to Yes/No using 0.5 threshold
y_pred_prob = model.predict(X_test)
y_pred      = (y_pred_prob > 0.5).astype(int)

print("\n── Confusion Matrix ──")
print(confusion_matrix(y_test, y_pred))

print("\n── Detailed Evaluation ──")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
