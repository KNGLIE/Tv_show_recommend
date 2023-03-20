# Import necessary modules
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import data_processing


# Define a class to represent a rule in the expert system
class Rule:
    def __init__(self, antecedent, consequence, weight):
        # Store the antecedent, consequence, and weight of the rule
        self.antecedent = antecedent
        self.consequence = consequence
        self.weight = weight


# Define a class to represent the expert system
# Define a class to represent the expert system
class ExpertSystem:
    def __init__(self, model, similarity_scores, rules):
        # Store the machine learning model, similarity scores, and rules for the expert system
        self.model = model
        self.similarity_scores = similarity_scores
        self.rules = rules

    def get_recommendations(self, tv_show):
            # Find the index of the input TV show in the dataset
            tv_show_index = new_df[new_df['title'] == tv_show].index[0]
            # Get the cluster label of the input TV show
            cluster_label = self.model.predict(vectors[tv_show_index].reshape(1, -1))[0]
            # Get the indices of all TV shows in the same cluster as the input TV show
            cluster_indices = list(filter(lambda i: (self.model.predict(vectors[i].reshape(1, -1)) == cluster_label).all(), range(len(new_df))))
            # Get the similarity scores between the input TV show and all other TV shows in the same cluster
            scores = self.similarity_scores[tv_show_index, cluster_indices]
            # Sort the similarity scores in descending order and return the top 10 TV shows
            top_indexes = sorted(range(len(cluster_indices)), key=lambda i: scores[i], reverse=True)[:10]
            return new_df.iloc[[cluster_indices[i] for i in top_indexes]]['title'].tolist()


# Prepare the TV show dataset
new_df = data_processing.prepare_data()

# Get the vector representation of all TV shows in the dataset using the TF-IDF technique
tfidf = TfidfVectorizer(max_features=8, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()

# Cluster the TV shows using the k-means algorithm
kmeans = KMeans(n_clusters=50)
kmeans.fit(vectors)

# Get the similarity scores between all TV shows in the dataset
similarity_scores = cosine_similarity(vectors)

# Define the rules for the expert system based on the cluster labels
rules = []
for i in range(50):
    rules.append(Rule(
        antecedent=lambda x: kmeans.predict(vectors[x].reshape(1, -1))[0] == i,
        consequence=1,
        weight=1
    ))

# Define a machine learning model for transforming TV show vectors
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu')
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(vectors, vectors, epochs=10, batch_size=32, verbose=0)