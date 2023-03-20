import data_processing
import similarity
from rules import ExpertSystem, rules, model, vectors
from sklearn.metrics.pairwise import cosine_similarity



# Compute the similarity scores between all TV shows in the dataset
similarity_scores = cosine_similarity(vectors)

# Create an expert system object
expert_system = ExpertSystem(model, similarity_scores, rules)

# Ask the user for the TV show they have recently watched
user = input('What TV Show have you recently watched?')

# Recommend a TV show to the user
recommendations = expert_system.get_recommendations(user)

# Print the recommended TV shows
print(recommendations)