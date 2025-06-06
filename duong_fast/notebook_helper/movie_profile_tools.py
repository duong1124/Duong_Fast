import pandas as pd
import re 
import random
from tqdm import tqdm

class MovieMetadata:
    def __init__(self, tmdb_metadata_path, df_genome_scores, df_genome_tags):
        self.movie_profile = pd.read_csv(tmdb_metadata_path, index_col = 'movieId') # DataFrame

        self.df_genome_scores = df_genome_scores 

        self.df_genome_tags = df_genome_tags
        self.tag_mapping = dict(zip(self.df_genome_tags["tagId"], self.df_genome_tags["tag"]))

    def check_null(self):
        for col in self.movie_profile.columns:
            null_count = self.movie_profile[col].isnull().sum()
            print(f"Column '{col}': {null_count} null values")

    def _get_title(self, movie_id):
        title = self.movie_profile.loc[movie_id, 'title']
        return title

    def _get_genres(self, movie_id):
        genres_list = self.movie_profile.loc[movie_id, 'genres'].split('|')
        return ", ".join(genres_list)  # Join genres with commas

    def _get_actors(self, movie_id):
        actors_str = self.movie_profile.loc[movie_id, 'actors']
        if pd.isnull(actors_str) or actors_str == '':
            return ""
        return actors_str

    def _get_director(self, movie_id):
        director_str = self.movie_profile.loc[movie_id, 'director']
        if pd.isnull(director_str) or director_str == '':
            return ""
        return director_str

    def _get_year(self, movie_id):
        title = self.movie_profile.loc[movie_id, 'title']
        match = re.search(r"\((\d{4})\)", title)  # Find year in parentheses
        if match:
            return match.group(1)  # Return the year if found
        else:
            return ""
    
    def get_paragraph(self, movie_id):
        year = self._get_year(movie_id)
        title = self._get_title(movie_id)
        genres = self._get_genres(movie_id)
        actors = self._get_actors(movie_id)
        director = self._get_director(movie_id)

        paragraph = f"Title: {title}. Year: {year}. Genres: {genres}. Actors: {actors}. Director: {director}."
        return paragraph
        
    def get_data_metadata(self, movieId, print_check = False):

        title = self._get_title(movieId)
        genres = self._get_genres(movieId)
        actors = self._get_actors(movieId)
        directors = self._get_director(movieId)
        year = self._get_year(movieId)

        if print_check:
            print(f"Movie ID: {movieId}")
            print(f"Title: {title}")
            print(f"Year: {year}")
            print(f"Genres: {genres}")
            print(f"Actors: {actors}")
            print(f"Director: {directors}")

        return title, genres, actors, directors, year

    def build_movie_profile(self, movie_id, num_tags=30):
        """
        Returns:
            dict
        """
        try:
            title, genres, actors, directors, year = self.get_data_metadata(movie_id)

            # genome tags
            genome_scores = self.df_genome_scores[self.df_genome_scores["movieId"] == movie_id]
            top_tags = genome_scores.sort_values(by="relevance", ascending=False).head(num_tags)

            genome_tags = [
            (self.tag_mapping[tag_id], relevance)
            for tag_id, relevance in zip(top_tags["tagId"], top_tags["relevance"])
            ]

            tags_scores = ", ".join([f"{tag} : {relevance:.4f}" for tag, relevance in genome_tags])
            movie_profile = {
                "Title": title,
                "Year": year,
                "Genres": genres,
                "Actors": actors,
                "Directors": directors,
                "Top tags with relevance scores": tags_scores,
            }
            return movie_profile
        
        except KeyError:
            print(f"Movie with ID {movie_id} not found in metadata.")
            return None

class CleanedMovieProfile(MovieMetadata):
    def __init__(self, tmdb_metadata_path, movie_profile_csv, df_genome_scores, df_genome_tags, df_ratings, num_tags):
        super().__init__(tmdb_metadata_path, df_genome_scores, df_genome_tags)
        if movie_profile_csv is not None:
            self.movie_profile = movie_profile_csv
        self.df_ratings = df_ratings
        self.num_tags = num_tags
        self.movieIds = list(df_ratings['movieId'].unique())
        self.cleaned_movie_profile = pd.DataFrame(index=self.movieIds, columns=['ProfileParagraph'])
        self.cleaned_movie_profile = self._profile_to_paragraph()

    def __repr__(self):
        return repr(self.cleaned_movie_profile)

    def _get_tags(self):
        tags_scores = {}
        for movieId in self.movieIds:
            genome_scores = self.df_genome_scores[self.df_genome_scores["movieId"] == movieId]
            top_tags = genome_scores.sort_values(by="relevance", ascending=False).head(self.num_tags)
            genome_tags = [
                (self.tag_mapping[tag_id], relevance)
                for tag_id, relevance in zip(top_tags["tagId"], top_tags["relevance"])
            ]

            text_tags_score = ", ".join([f"{tag} : {relevance:.4f}" for tag, relevance in genome_tags])
            tags_scores[movieId] = text_tags_score
        return tags_scores

    def _profile_to_paragraph(self, using_description=False):
        tags_scores = self._get_tags()
        for movieId in tqdm(self.movieIds, desc="Creating movie paragraphs", unit="movie"):
            paragraph = self.get_paragraph(movieId)
            paragraph += f" TopTagsWithScores: {tags_scores[movieId]}"

            if using_description:
                description = self.movie_profile.loc[movieId, 'description']
                if pd.notnull(description) and description != '':
                    paragraph += f" Description: {description}"
                    
            self.cleaned_movie_profile.loc[movieId, 'ProfileParagraph'] = paragraph
            
        return self.cleaned_movie_profile

    def print_a_profile(self):
        movieId = random.choice(self.movieIds)
        print(f"movieId: {movieId}")
        print(self.cleaned_movie_profile.loc[movieId, 'ProfileParagraph'])
