import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
import time

def clean_text(html: str) -> str:
    """
    Returns a processed string that removes all html segments.

    Parameters:
    - html (str): The html code in a string format.

    Returns:
    str: a processsed string
    """
    
    #defining soup obj
    soup = BeautifulSoup(html, 'html.parser')

    return soup.get_text()


def LSA(html: str, dbug: bool = False) -> None:
    """
    Performs Latent Semantic Analysis (LSA) on an HTML document.

    Parameters:
    - html (str): The HTML code to be processed.
    - dbug (bool): Debug mode. If True, print additional information.

    Returns:
    None
    """
    # Clean the HTML document
    cleaned_html = clean_text(html)
    
    if dbug:
        print(cleaned_html)

    # Define a stop set
    stopset = stopwords.words('english')

    # Define a vectorizer
    vectorizer = TfidfVectorizer(stop_words=stopset, use_idf=True, ngram_range=(1, 3))

    # Define the sparse matrix
    X = vectorizer.fit_transform([cleaned_html])

    # Define the decomposer
    lsa = TruncatedSVD(n_components=27, n_iter=300)
    lsa.fit(X)

    # Get feature names and weights
    terms = vectorizer.get_feature_names_out()
    term_weights = list(zip(terms, lsa.components_[0]))

    # Sort terms based on weights in descending order
    sorted_terms = sorted(term_weights, key=lambda x: x[1], reverse=True)

    # Print the most important words
    print('Most important words (sorted by weight):')
    for term, weight in sorted_terms:
        print(f'{term}: {weight}')

    # Print the sparse matrix if in debug mode
    if dbug:
        print('Sparse Matrix:')
        print(X)
        print('Sparse Matrix Shape:', X.shape)
        print('LSA Components:', lsa.components_[0])

