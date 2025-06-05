from pydantic import BaseModel
import itertools
import time
import json
from tqdm import tqdm
import pandas as pd
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
        if self.time_th is None:
            raise ValueError("We are not generating user profiles here, or you forgot.")
        
        if time_th is not None:
            self.time_th = time_th

        batch_size = len(self.userId_train) // num_splits
        start_idx = (self.time_th - 1) * batch_size
        if self.time_th == num_splits:
            userId_gen = self.userId_train[start_idx:]
        else:
            end_idx = self.time_th * batch_size
            userId_gen = self.userId_train[start_idx:end_idx]
        return userId_gen
    
    def all_userId_gen(self):
        remainder = (self.time_th % 10) if (self.time_th % 10) != 0 else 10

        for i in range(remainder, 101, 10):
              print(f"{i}_th time: {self.split_userId_train(i, num_splits=100)} ")


    def generate_user_profile(self, user_id, api_key, nap = True, prompt_template=None):
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
                time.sleep(30)
            return user_id, {"error": f"Failed to generate profile: {str(e)}"}

    def build_user_profiles(self, userId_gen = None, nap = True):
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