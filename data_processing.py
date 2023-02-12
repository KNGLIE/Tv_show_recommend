import pandas as pd
from nltk import PorterStemmer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


    # This line is to avoid the SettingWithCopyWarning
def prepare_data():
    # Read the csv file
    pd.options.mode.chained_assignment = None
    # Select the columns that we need
    tv_show_db = pd.read_csv('tv-db.csv')
    tv_show_db = tv_show_db[
    # Check for null values
        ['id', 'title', 'genre', 'description', 'IMDb', 'Rotten Tomatoes', 'Netflix', 'Hulu', 'Prime Video', 'Disney+']]
    # Drop the null values
    tv_show_db.isnull().sum()
    # Drop the duplicates
    tv_show_db.dropna(inplace=True)
    # Split the description column
    tv_show_db.drop_duplicates('title', keep='first', inplace=True)
    # Remove the spaces from the genre column
    tv_show_db['description'] = tv_show_db['description'].apply(lambda x: x.split())
    # Remove the spaces from the description column
    tv_show_db['genre'] = tv_show_db['genre'].apply(lambda x: [i.replace(" ", "") for i in x])
    # Create a new column called tags which is the combination of description and genre
    tv_show_db['description'] = tv_show_db['description'].apply(lambda x: [i.replace(" ", "") for i in x])
    # Select the columns that we need
    tv_show_db['tags'] = tv_show_db['description'] + tv_show_db['genre']
    # Join the tags column
    new_df = tv_show_db[['id', 'title', 'tags', 'IMDb', 'Rotten Tomatoes']]
    # Convert the tags column to lower case
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    # Convert the title column to lower case
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    # Stem the tags column
    new_df['title'] = new_df['title'].apply(lambda x: x.lower())
    new_df['tags'] = new_df['tags'].apply(stem)
    return new_df


ps = PorterStemmer()
    # Stem the text


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


