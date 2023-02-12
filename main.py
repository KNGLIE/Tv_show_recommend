import data_processing
import similarity
from rules import ExpertSystem, rules, model


# Ask the user for the TV show they have recently watched
user = input('What TV Show have you recently watched?')

# Prepare the data for the similarity computation
tv_df = data_processing.prepare_data()

# Compute the similarity scores between the user's TV show and all the other TV shows
similarity_scores = similarity.compute_similarity(tv_df)

# Create an expert system object
expert_system = ExpertSystem(rules, model)

# Recommend a TV show to the user
expert_system.recommend_tv_show(user)
