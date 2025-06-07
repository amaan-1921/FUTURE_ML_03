import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import numpy as np
from preprocess import clean_text

# Create model directory
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Load customer support tickets
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
df = pd.read_csv(os.path.join(data_dir, 'customer_support_tickets.csv'))
df = df[['Ticket Description', 'Ticket Type']].dropna()
df.rename(columns={'Ticket Description': 'text', 'Ticket Type': 'intent'}, inplace=True)

# Load intents.json
intents_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'intents.json')
with open(intents_path, 'r') as f:
    intents_data = json.load(f)

# Prepare intents.json data (oversample 20 times)
queries = []
intents = []
for _ in range(20):  # Oversample to counter imbalance
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            queries.append(pattern)
            intents.append(intent['tag'])
intents_df = pd.DataFrame({'text': queries, 'intent': intents})

# Combine datasets
df = pd.concat([df, intents_df], ignore_index=True)

# Clean text
df['cleaned_text'] = df['text'].apply(clean_text)

# Remove empty rows
df = df[df['cleaned_text'] != '']

# Features and labels
X = df['cleaned_text']
y = df['intent']

# Print class distribution
print("Intent distribution:\n", y.value_counts())

# Compute class weights
class_counts = y.value_counts().to_dict()
total_samples = len(y)
class_weights = {label: total_samples / (len(class_counts) * count) for label, count in class_counts.items()}
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_weights_dict = {i: class_weights[label] for i, label in enumerate(label_encoder.classes_)}

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Encode labels for training
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Build TensorFlow model
model = Sequential([
    Dense(64, input_dim=X_train_vec.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train with class weights
model.fit(X_train_vec, y_train_encoded, epochs=30, batch_size=16, class_weight=class_weights_dict, verbose=1)

# Evaluate
y_pred_encoded = np.argmax(model.predict(X_test_vec), axis=1)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Show misclassified examples
print("\nSome misclassified samples:")
misclassified_mask = y_test != y_pred
misclassified_texts = X_test[misclassified_mask]
misclassified_true = y_test[misclassified_mask]
misclassified_pred = [pred for pred, mask in zip(y_pred, misclassified_mask) if mask]
for i, text in enumerate(misclassified_texts.head(5)):
    print(f"\nText: {text}")
    print(f"True: {misclassified_true.iloc[i]}")
    print(f"Predicted: {misclassified_pred[i]}")

# Save model, vectorizer, and encoder
model.save(os.path.join(model_dir, 'intent_classifier.h5'))
joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))