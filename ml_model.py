import numpy as np
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Define model paths
MODEL_DIR = 'models'
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, 'fasttext_model.bin')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# Load the models
try:
    ft_model = FastText.load(FASTTEXT_MODEL_PATH)
    
    with open(LR_MODEL_PATH, 'rb') as f:
        lr_model = pickle.load(f)
        
    with open(VECTORIZER_PATH, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    ft_model = None
    lr_model = None
    tfidf_vectorizer = None

def simple_vectorize_text(text):
    """
    Fallback method to vectorize text using simple averaging of word vectors
    """
    words = text.lower().split()
    word_vectors = [ft_model.wv[word] for word in words if word in ft_model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    return np.zeros(ft_model.vector_size)

def vectorize_text(text):
    """
    Vectorize a single text review using FastText embeddings weighted by TF-IDF
    """
    try:
        words = text.lower().split()
        tfidf_weights = tfidf_vectorizer.transform([text]).toarray()[0]
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        weighted_vectors = []
        for word in words:
            if word in feature_names:
                try:
                    word_vector = ft_model.wv[word]
                    word_weight = tfidf_weights[feature_names.tolist().index(word)]
                    weighted_vectors.append(word_vector * word_weight)
                except KeyError:
                    continue
                    
        if weighted_vectors:
            return np.mean(weighted_vectors, axis=0)
        return np.zeros(ft_model.vector_size)
    except Exception as e:
        print(f"Error in TF-IDF vectorization: {str(e)}")
        return simple_vectorize_text(text)

def generate_recommendation(review_text):
    """
    Generate recommendation (0 or 1) based on review text
    """
    try:
        if ft_model is None or lr_model is None:
            raise ValueError("Models not properly loaded")

        # Vectorize the review text
        review_vector = vectorize_text(review_text)
        
        # Reshape for prediction
        review_vector = review_vector.reshape(1, -1)
        
        # Predict using logistic regression model
        prediction = lr_model.predict(review_vector)[0]
        
        return int(prediction)
    
    except Exception as e:
        print(f"Error generating recommendation: {str(e)}")
        # Fallback to a simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'love', 'like', 'recommend']
        negative_words = ['bad', 'poor', 'terrible', 'hate', 'dislike', 'not recommend']
        
        review_words = review_text.lower().split()
        positive_count = sum(1 for word in review_words if word in positive_words)
        negative_count = sum(1 for word in review_words if word in negative_words)
        
        return 1 if positive_count >= negative_count else 0