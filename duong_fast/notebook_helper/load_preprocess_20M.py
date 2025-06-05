import os
import urllib.request
import zipfile
import pandas as pd

class LoadPreprocess20M:
    def dataload(self, dataset_url="http://files.grouplens.org/datasets/movielens/ml-20m.zip", dataset_path="ml-20m.zip", extract_path="ml-20m"):
        """
        Download and extract the MovieLens 20M dataset, then load all CSV files into pandas DataFrames.
        Returns: df_movies, df_genome_scores, df_tags, df_genome_tags, df_ratings, df_links
        """
        if not os.path.exists(dataset_path):
            urllib.request.urlretrieve(dataset_url, dataset_path)

        if not os.path.exists(extract_path):
            with zipfile.ZipFile(dataset_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

        base = os.path.join(extract_path, "ml-20m")

        df_movies = pd.read_csv(os.path.join(base, "movies.csv"))
        df_genome_scores = pd.read_csv(os.path.join(base, "genome-scores.csv"))
        df_tags = pd.read_csv(os.path.join(base, "tags.csv"))
        df_genome_tags = pd.read_csv(os.path.join(base, "genome-tags.csv"))
        df_ratings = pd.read_csv(os.path.join(base, "ratings.csv"))
        df_links = pd.read_csv(os.path.join(base, "links.csv"))

        return df_movies, df_genome_scores, df_tags, df_genome_tags, df_ratings, df_links

    def _preprocess_df_tags(self, df_tags):
        return df_tags # No specific preprocessing atm
    
    def _preprocess_df_links(self, df_links):
        return df_links # No specific preprocessing atm
    
    def _preprocess_df_genome_tags(self, df_genome_tags):
        return df_genome_tags # No specific preprocessing atm
    
    def preprocess(self, df_movies, df_genome_scores, df_tags, df_genome_tags, df_ratings, df_links, tags_links_pp = False):
        """
        Preprocess the loaded DataFrames: filter movies and users with at least 20 ratings and only keep movies with genome tags.
        Args:
            6 DataFrames of MovieLens 20M
            tags_links_pp: boolean, if True, preprocess tags, links and genome tags
        Returns: 
            (final_movies, final_genome_scores, final_ratings, tag_mapping) if tags_links_pp is False
            Else *all
        """
        # drop out the movies which do not have tag genome
        moviesId_with_genome = df_genome_scores['movieId'].unique()
        df_movies_with_genome = df_movies[df_movies['movieId'].isin(moviesId_with_genome)]
        df_ratings_genome = df_ratings[df_ratings['movieId'].isin(moviesId_with_genome)]

        # only movies and users with at least 20 ratings are kept

        # count ratings
        movie_ratings_count = df_ratings['movieId'].value_counts()
        user_ratings_count = df_ratings['userId'].value_counts()
        # movie and user Id with more than 20 ratings
        moviesId_more_20_ratings = movie_ratings_count[movie_ratings_count >= 20].index
        usersId_more_20_ratings = user_ratings_count[user_ratings_count >= 20].index
        
        final_ratings = df_ratings_genome[df_ratings_genome['movieId'].isin(moviesId_more_20_ratings) & df_ratings_genome['userId'].isin(usersId_more_20_ratings)]
        final_movies = df_movies_with_genome[df_movies_with_genome['movieId'].isin(moviesId_more_20_ratings)]
        final_genome_scores = df_genome_scores[df_genome_scores['movieId'].isin(final_movies['movieId'])]

        tag_mapping = dict(zip(df_genome_tags["tagId"], df_genome_tags["tag"]))

        if tags_links_pp:
            final_tags = self._preprocess_df_tags(df_tags)
            final_links = self._preprocess_df_links(df_links)
            final_genome_tags = self._preprocess_df_genome_tags(df_genome_tags)
            return final_movies, final_genome_scores, final_tags, final_genome_tags, final_ratings, final_links, tag_mapping
        
        return final_movies, final_genome_scores, final_ratings, tag_mapping