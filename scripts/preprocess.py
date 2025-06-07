import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define contraction mapping
contraction_mapping = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "didn't": "did not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "cannot": "can not"
}

# Keep negations and key terms
negations = {"no", "not", "do", "does", "did", "can", "cannot", "won't", "should", "could", "login", "account"}
stop_words = set(stopwords.words('english')) - negations

lemmatizer = WordNetLemmatizer()

def expand_contractions(text):
    for contraction, expanded in contraction_mapping.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r'[^a-z\s\?]', '', text)  # Keep alphabets, spaces, question marks
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens) if tokens else ''