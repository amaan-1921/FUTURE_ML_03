# FUTURE_ML_03
## Task 3: Build a Chatbot for Customer Support

This repository contains the code for building a simple chatbot that can respond to customer queries using an FAQ dataset.

## Features

- Intent classification using a deep learning model (`intent_classifier.h5`)
- Text preprocessing and vectorization
- Multiple predefined intents with sample patterns and responses (stored in `intents.json`)
- Flask-based web interface for interactive chatting
- Logs predicted intents for future insights

## Notes
- The chatbot uses a trained Keras model for intent classification.
- Text preprocessing is handled in scripts/preprocess.py.
- Responses are fetched from intents.json based on predicted intent.
- Intent logs are saved to intent_logs.txt for future analysis.

### Dataset

- Link to dataset: [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/waseemalastal/customer-support-ticket-dataset)
