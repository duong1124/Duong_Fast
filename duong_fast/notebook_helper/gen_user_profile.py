from pydantic import BaseModel
import itertools
import time
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from notebook_helper import MovieMetadata

class TagScore(BaseModel):
    tag: str
    score: float

class UserProfile(BaseModel):
  FavoriteDecade: str
  FavoriteGenres: list[str]
  FavoriteActors: list[str]
  FavoriteDirectors: list[str]
  TopTagsWithScores: list[TagScore]

class GenUserProfile(MovieMetadata):
    """
    Class to generate user profiles based on their movie ratings.
    Inherits from MovieMetadata to access build_movie_profile.
    """

    def __init__(self, tmdb_metadata_path, rating_train, df_genome_scores, df_genome_tags, num_top_movies = 25, num_top_tags = 60, time_th = None, api_keys = None, gen_users_path = None):
        super().__init__(tmdb_metadata_path, df_genome_scores, df_genome_tags)
        self.rating_train = rating_train
        self.num_top_movies = num_top_movies
        self.num_top_tags = num_top_tags
        self.time_th = time_th
        self.api_keys = api_keys
        self.gen_users_path = gen_users_path
        self.userId_train = rating_train['userId'].unique()
    
    def split_userId_train(self, time_th = None, num_splits=100):
        if self.time_th is None and time_th is None:
            raise ValueError("We are not generating user profiles here, or you forgot.")
        
        # Avoid modifying the original self.time_th
        th = time_th if time_th is not None else self.time_th

        batch_size = len(self.userId_train) // num_splits
        start_idx = (th - 1) * batch_size
        if th == num_splits:
            userId_gen = self.userId_train[start_idx:]
        else:
            end_idx = th * batch_size
            userId_gen = self.userId_train[start_idx:end_idx]
        return userId_gen
    
    def all_userId_gen(self):
        remainder = (self.time_th % 10) if (self.time_th % 10) != 0 else 10

        for i in range(remainder, 101, 10):
              print(f"{i}_th time: {self.split_userId_train(i, num_splits=100)} ")


    def generate_user_profile(self, user_id, api_key, nap = 15, prompt_template=None):
        user_ratings = self.rating_train[self.rating_train['userId'] == user_id]

        # Sort by rating and limit to top movies if specified
        num_movies = len(user_ratings)
        top_movies = min(num_movies, self.num_top_movies)
        user_ratings = user_ratings.sort_values(by='rating', ascending=False).head(top_movies)

        # Build movie profiles for the user's rated movies
        movie_profiles = []
        for _, row in user_ratings.iterrows():
            movie_id = row['movieId']
            rating = row['rating']

            # Get movie profile using the provided function
            movie_profile = self.build_movie_profile(movie_id, num_tags=self.num_top_tags)
            if movie_profile:
                movie_profile['User Rating'] = rating
                movie_profiles.append(movie_profile)

        if prompt_template is not None:
            prompt = prompt_template.format(
                user_id=user_id,
                num_movies=len(movie_profiles),
                movie_profiles=movie_profiles,
                num_tags=self.num_top_tags
            )
        else:
            prompt = (
                "You are an expert data analyst specializing in movie recommendation systems.\n"
                "Analyze the following user preferences based on the movies and ratings they have given.\n\n"
                f"User ID: {user_id}\n"
                f"Total Movies Rated: {len(movie_profiles)}\n\n"
                "Movie Profiles with Ratings:\n\n"
            )

            # Add each movie profile to the prompt
            for profile in movie_profiles:
                prompt += f"Movie: {profile['Title']}\n"
                prompt += f"Rating: {profile['User Rating']}\n"
                prompt += f"Year: {profile['Year']}\n"
                prompt += f"Genres: {profile['Genres']}\n"
                prompt += f"Actors: {profile['Actors']}\n"
                prompt += f"Directors: {profile['Directors']}\n"
                prompt += f"Top tags: {profile['Top tags with relevance scores']}\n\n"

            prompt += (
                "Guidelines:\n"
                "1. Analyze the user's movie ratings and generate a profile of their preferences.\n"
                "2. 10 Favorite Actors and 5 Favorite Directors, they should come from highly rating movies. \n"
                "3. For TopTagsWithScores, analyze all movies the user has watched and calculate tag scores based on:\n"
                " How often a tag appears across the user's watched movies\n"
                " The relevance score of each tag for each movie\n"
                " The user's rating for each movie (higher ratings should give tags more weight)\n"
                " Normalize final scores to a 0-1 scale (same as Top tags of the Movie Profiles)\n"
                f" Number of top tags should equal {self.num_top_tags}, same as number of tags with a movie\n\n"
                "Return a user profile in this JSON format:\n"
                "{\n"
                '  "FavoriteDecade": "YYYY",\n'
                '  "FavoriteGenres": ["Genre1", "Genre2", ...],\n'
                '  "FavoriteActors": ["Actor1", "Actor2", ...],\n'
                '  "FavoriteDirectors": ["Director1", "Director2", ...],\n'
                '  "TopTagsWithScores": [{"Tag1": N}, {"Tag2": N}, ...]\n'
                "}\n"
            )

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type='application/json',
                    response_schema=UserProfile,
                )
            )

            # Parse the response
            response_text = response.text
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            user_profile = json.loads(response_text)
            return user_id, user_profile

        except Exception as e:
            if nap:
                time.sleep(nap)
            return user_id, {"error": f"Failed to generate profile: {str(e)}"}

    def build_user_profiles(self, userId_gen = None, nap = 15):
        if userId_gen is None:
            userId_gen = self.split_userId_train()

        user_profiles = {}
        api_key_cycle = itertools.cycle(self.api_keys) # Create a cycle of API keys

        for user_id in tqdm(userId_gen, desc="Building user profiles", unit="user"):
            current_api_key = next(api_key_cycle) # Get the next API key from the cycle
            user_id, profile = self.generate_user_profile(user_id, current_api_key, nap = nap)
            user_profiles[user_id] = profile
            time.sleep(4)  # Rate limit for API calls

        return user_profiles

    def save_user_profiles(self, user_profiles, gen_users_path = None):
        if gen_users_path is None:
            gen_users_path = self.gen_users_path

        filename = f"user_1384_{self.time_th}_{self.num_top_movies}_{self.num_top_tags}.json"
        user_profiles_serializable = {int(k): v for k, v in user_profiles.items()}

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(user_profiles_serializable, f, indent=2, ensure_ascii=False)

        os.makedirs(gen_users_path, exist_ok=True)

        # Define the file path within the folder
        file_path = os.path.join(gen_users_path, filename)

        # Save the JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(user_profiles_serializable, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {file_path}")

    def refill_user_profiles_and_save_json(self, df_users_main, nan_users,
                                           max_retries=5, attempt=0):
        
        # exit if reached max_retries
        if attempt >= max_retries:
            print(f"Maximum number of retries ({max_retries}) reached. Exiting.")
            print(f"Number of users with NaN values: {len(nan_users)}")
            return df_users_main
        
        # base case - as no more NaN users
        if len(df_users_main.columns[df_users_main.isnull().any()]) == 0:
            print("Completed refilling user profiles")
            return df_users_main
        
        print(f"Attempt {attempt + 1}: Rebuilding {len(nan_users)} users...")
        refilled = self.build_user_profiles(nan_users)
        df_refilled = pd.DataFrame(refilled)

        if df_refilled.shape[0] == 6:
            print(f"Attempt {attempt + 1} still has nan users due to existence of error row.")
            df_refilled = df_refilled.drop(df_refilled.index[-1])

        new_nan_users = df_refilled.columns[df_refilled.isnull().any()].tolist()
        print(f"Attempt {attempt + 1} still has {len(new_nan_users)} NaN users")

        # Save refill file into json, avoid using to_csv function as it affects datatype
        json_filename = f"refilled_profiles_attempt_{attempt + 1}.json"
        json_file_path = os.path.join(self.gen_users_path, json_filename)
        with open(json_file_path, "w") as json_file:
            json.dump(refilled, json_file)

        print(f"Saved refilled profiles to {json_file_path}")

        df_filled = df_users_main.combine_first(df_refilled)

        # recursion call
        return self.refill_user_profiles_and_save_json(
            df_filled,
            new_nan_users,
            max_retries=max_retries,
            attempt=attempt + 1
        )
    
    @staticmethod
    # Incase the tag_scores that LLM generate is not good, we will consider assigning user favorite tags with revelance scores as average of top movies they watched
    def calculate_user_genome_scores(df_ratings, df_genome_scores, userIds):
        # if we don't take subset, we take all users
        if userIds is None:
            userIds = df_ratings['userId'].unique()
        n_users = len(userIds)

        df_genome_scores_pivot = df_genome_scores.pivot(index='movieId', columns='tagId', values='relevance')

        dim_genome = df_genome_scores_pivot.shape[1]
        user_genome_matrix = np.zeros((n_users, dim_genome))

        for i, userId in enumerate(userIds):
            user_ratings = df_ratings[df_ratings['userId'] == userId]

            # movie_indices = user_ratings[:, 1]
            # ratings = user_ratings[:, 2]

            movie_indices = user_ratings['movieId'].values.astype(int)
            ratings = user_ratings['rating'].values

            normalized_ratings = ratings / 5.0

            user_genome_matrix[i] = np.sum(normalized_ratings[:, None] * df_genome_scores_pivot.loc[movie_indices], axis=0) / len(movie_indices)

        return user_genome_matrix
    
    @staticmethod
    def detect_json_errors(df: pd.DataFrame, output_path: str) -> list[str]:
        """
        Detect errors in JSON data loaded into a DataFrame, remove the erroneous user entries,
        and save the cleaned data to a new JSON file.

        Parameters:
        - df: DataFrame with user IDs as index and columns for user data.
        - output_path: Path to save the cleaned JSON data.

        Returns:
        - List of user IDs with errors (missing keys, empty lists, or invalid types).
        """
        expected_keys = {
            'FavoriteDecade': str,
            'FavoriteGenres': list,
            'FavoriteActors': list,
            'FavoriteDirectors': list,
            'TopTagsWithScores': list
        }

        error_ids = []

        # Check for missing columns
        missing_columns = [key for key in expected_keys if key not in df.columns] # check if expected_keys not in cols, if yes -> add to list
        if missing_columns:
            error_ids = df.index.astype(str).tolist()
            cleaned_df = df.drop(index=error_ids)
            cleaned_df.to_json(output_path, orient='index', indent=2)
            return sorted(list(set(error_ids)))

        for user_id, row in df.iterrows():
            for key, expected_type in expected_keys.items():
                value = row.get(key)
                # na
                if pd.isna(value):
                    error_ids.append(str(user_id))
                    break
                # wrong type
                if not isinstance(value, expected_type):
                    error_ids.append(str(user_id))
                    break
                # if expected_type = list but empty
                if expected_type == list and not value:
                    error_ids.append(str(user_id))
                    break

                if key == 'TopTagsWithScores':
                    if not all(
                        isinstance(item, dict) and
                        isinstance(item.get('tag'), str) and
                        isinstance(item.get('score'), (int, float))
                        for item in value
                    ):
                        error_ids.append(str(user_id))
                        break

        # Remove error rows from the DataFrame
        cleaned_df = df[~df.index.astype(str).isin(error_ids)]

        # Save cleaned data to JSON
        cleaned_df.to_json(output_path, orient='index', indent=2)

        return sorted(list(set(error_ids)))
    
    @staticmethod
    def combine_and_sort_json_files(folder_path):
        # Initialize an empty list to store DataFrames
        dfs = []

        # Ensure the folder path exists
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"The path {folder_path} is not a valid directory")

        # Iterate through all JSON files in the folder
        for file_path in folder.glob("*.json"):
            try:
                # Read JSON file into a DataFrame with user IDs as index
                data = pd.read_json(file_path, orient='index')
                dfs.append(data)
            except ValueError as e:
                print(f"Error decoding JSON in {file_path}: {e}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if not dfs:
            raise ValueError("No valid JSON files found in the directory")

        # Concatenate all DataFrames
        combined_df = pd.concat(dfs)

        # Reset index to make user ID a column, then sort by user ID
        combined_df = combined_df.reset_index().rename(columns={'index': 'userId'})
        combined_df['userId'] = combined_df['userId'].astype(int)  # Ensure userId is numeric
        sorted_df = combined_df.sort_values(by='userId')

        # Set userId back as the index for JSON output
        sorted_df = sorted_df.set_index('userId')

        # Save the combined and sorted DataFrame as a new JSON file in the same folder
        output_path = folder / "combined_sorted_users_pandas.json"
        try:
            sorted_df.to_json(output_path, orient='index', indent=4)
            print(f"Combined and sorted data saved to {output_path}")
        except Exception as e:
            print(f"Error saving combined data to {output_path}: {e}")

        return sorted_df