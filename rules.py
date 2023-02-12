import os
import sys
import logging
logging.basicConfig(level=logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import data_processing
from similarity import get_vectors
tf.compat.v1.logging.set_verbosity(v=1)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Rule:
    def __init__(self, antecedent, consequence, weight):
        self.antecedent = antecedent
        self.consequence = consequence
        self.weight = weight


class ExpertSystem:
    def __init__(self, rules, model):
        self.rules = rules
        self.model = model

    def get_recommendations(self, tv_show):

        tv_show_index = new_df[new_df['title'] == tv_show].index[0]
        tv_show_vector = vectors[tv_show_index].reshape(1, -1)
        tv_show_vector = self.model.predict(tv_show_vector, callbacks=None, verbose=None)

        similarity_scores = []

        for i, show in enumerate(vectors):

            if i != tv_show_index:
                show = show.reshape(1, -1)
                show = self.model.predict(show, callbacks=None, verbose=None)
                score = cosine_similarity(tv_show_vector, show)[0][0]
                similarity_scores.append((i, score))
        return sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    def recommend_tv_show(self, tv_show):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        recommendations = self.get_recommendations(tv_show)
        if not recommendations:
            print("No recommendations found")
            return
        for recommendation in recommendations[:10]:
            print(new_df.iloc[recommendation[0]].title)


rules = [
    Rule(
        antecedent=lambda x: x in [0, 1, 2, 3],
        consequence=1,
        weight=1
    ),
    Rule(
        antecedent=lambda x: x in [4, 5, 6],
        consequence=0.5,
        weight=0.5
    ),
    Rule(
        antecedent=lambda x: x in [7, 8, 9, 10],
        consequence=0.1,
        weight=0.1
    ),
]

new_df = data_processing.prepare_data()
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu')
])
vectors = get_vectors()
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(vectors, vectors, epochs=10, batch_size=32, verbose=0)


def create_expert_system(rules, model):
    sys.stderr = open(os.devnull, 'w')
    return ExpertSystem(rules, model)
