import os
import urllib.request
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LoadPreprocess20M:
    def dataload(self, dataset_url="http://files.grouplens.org/datasets/movielens/ml-20m.zip", dataset_path="ml-20m.zip", extract_path="ml-20m", local=False):
        """
        Download and extract the MovieLens 20M dataset, then load all CSV files into pandas DataFrames.
        If local=True, load from local folder (ml-20m/ml-20m/) and skip download/extract.
        Returns: df_movies, df_genome_scores, df_tags, df_genome_tags, df_ratings, df_links
        """
        if local:
            base = os.path.join("ml-20m", "ml-20m")
            required_files = [
                "movies.csv", "genome-scores.csv", "tags.csv", "genome-tags.csv", "ratings.csv", "links.csv"
            ]
            missing = [f for f in required_files if not os.path.exists(os.path.join(base, f))]
            if missing:
                raise FileNotFoundError(f"Missing required files in {base}: {missing}\nPlease ensure all MovieLens 20M CSVs are present.")
        else:
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
    
    @staticmethod
    def draw_user_ratings_distribution(df_ratings, method='mean', plot_type='histogram', bins=50):
        """
        Plots the distribution of user ratings (mean/median) across all users from MovieLens 20M.

        Args:
            df_ratings (pd.DataFrame): DataFrame with 'userId' and 'rating' columns.
            method (str): Aggregation method ('mean' or 'median'). Defaults to 'mean'.
            percentiles (list): Percentiles to calculate. Defaults to [25, 50, 75].
            plot_type (str): Type of plot ('histogram', 'kde', or 'boxplot'). Defaults to 'histogram'.
            bins (int): Number of bins for the histogram. Defaults to 50.
        """

        if method not in ['mean', 'median']:
            raise ValueError("Method should be 'mean' or 'median'.")

        if method == 'mean':
            user_aggregated_ratings = df_ratings.groupby('userId')['rating'].mean()
            title_suffix = "Mean Rating"
        else:  # method == 'median'
            user_aggregated_ratings = df_ratings.groupby('userId')['rating'].median()
            title_suffix = "Median Rating"

        print(f"Descriptive for {method} ratings:\n{user_aggregated_ratings.describe()}")

        plt.figure(figsize=(12, 6))

        if plot_type == 'histogram':
            sns.histplot(user_aggregated_ratings, bins=bins, kde=True)
            plt.title(f'Distribution of User {title_suffix}', fontsize=16)
            plt.xlabel(f'User {title_suffix}', fontsize=12)
            plt.ylabel('Number of Users', fontsize=12)

        elif plot_type == 'kde':
            sns.kdeplot(user_aggregated_ratings, fill=True)
            plt.title(f'Kernel Density Estimate of User {title_suffix}', fontsize=16)
            plt.xlabel(f'User {title_suffix}', fontsize=12)
            plt.ylabel('Density', fontsize=12)

        elif plot_type == 'boxplot':
            plt.boxplot(user_aggregated_ratings, vert=False)
            plt.title(f'Box Plot of User {title_suffix}', fontsize=16)
            plt.xlabel(f'User {title_suffix}', fontsize=12)
            plt.yticks([])

        else:
            raise ValueError("Plot_type should be 'histogram', 'kde', or 'boxplot'.")

        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.show()