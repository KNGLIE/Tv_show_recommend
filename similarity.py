from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import data_processing


def compute_similarity(data):

    cosine_similarity(get_vectors())
    similarity = cosine_similarity(get_vectors())
    return similarity


def get_vectors():
    cv = CountVectorizer(max_features=8, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    return vectors


new_df = data_processing.prepare_data()