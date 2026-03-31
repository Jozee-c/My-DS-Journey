import pandas as pd
import re
import sys
import nltk

from contractions import fix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# 1. Load the data.
def load_data(file_path):
    """Load the data."""
    df = pd.read_csv(file_path, sep='\t', header=0, engine='python', quoting=3)
    texts = df['Sentence'].fillna('').tolist()
    return texts


# 2. Preprocessing text data.
def preprocess_text(texts):
    """
    Preprocess the input text.
        Steps: 1. Convert to lowercase
            2. Remove punctuation, numbers and special characters
            3. Normalisation
            4. Tokenization
            5. Remove stop words
            6. Lemmatization
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    extra_stop_words = ('thee', 'thou', 'thy', 'hath', 'doth', 'tis', 'art',
                        'hast', 'wherefore', 'thence', 'hither', 'ere', 'oft')

    cleaned_texts = []
    for text in texts:
        text = text.lower()   # Convert to lowercase
        text = text.replace("\\n", " ")   # Replace newline characters with space.
        text = fix(text)   # Fix contractions (e.g., "don't" to "do not") and typos。
        text = re.sub(r'[^a-z\s]', ' ', text)   # Remain only letters and whitespaces.
        tokens = text.split()   # Tokenize the text
        tokens = [word for word in tokens if word not in stop_words and
                  word not in extra_stop_words]   # Remove stop words
        tokens = [lemmatizer.lemmatize(word) for word in tokens]   # Lemmatization
        cleaned_texts.append(' '.join(tokens))

    return cleaned_texts


# 3. Vectorize using TF-IDF.
def vectorize(cleaned_texts, max_features=3000):
    """Extract features using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=2,
                                 max_df=0.85,
                                 sublinear_tf=True)
    X = vectorizer.fit_transform(cleaned_texts)
    return X


# 4. Dimensionality reduction and Normalization.
def reduce_and_normalize(X, n_components=100):
    """
    Apply Truncated SVD for dimensionality reduction
    and normalize feature vectors to unit length.
    """
    # Dimensionality reduction.
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)

    # Normalization.
    normalizer = Normalizer(copy=False)
    X_normalized = normalizer.fit_transform(X_reduced)

    return X_normalized


# Run standalone for a quick sanity check.
if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data_train.txt"
    texts = load_data(file_path)
    cleaned_texts = preprocess_text(texts)
    X = vectorize(cleaned_texts, max_features=3000)
    print(f"Matrix shape: {X.shape}")
    X_normalized = reduce_and_normalize(X, n_components=100)
    print(f"Matrix shape after reduction and normalization: {X_normalized.shape}")
    print(f"Sample sentence [0]: {cleaned_texts[0][:80]}")
