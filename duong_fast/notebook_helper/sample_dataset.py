import pandas as pd
import numpy as np

class SampleRatings:
    def __init__(self, df_ratings, sample_percent=1.0, sample_method='random', random_state=42):
        self.df_ratings = df_ratings
        self.sample_percent = sample_percent
        self.sample_method = sample_method
        self.random_state = random_state
    
    def sample_users(self):
        """
        Sample ratings by %users.

        Returns: sampled ratings, sampled userIds
        """
        userIds = pd.Series(self.df_ratings['userId'].unique())
        sample_size = int(len(userIds) * self.sample_percent)

        if self.sample_method == 'random':
            sample_user_ids = userIds.sample(n=sample_size, random_state=self.random_state)
        elif self.sample_method == 'first':
            sample_user_ids = userIds.iloc[:sample_size]
        else:
            raise ValueError("method must be 'random' or 'first'")
        
        gen_file = self.df_ratings[self.df_ratings['userId'].isin(sample_user_ids)]

        #print(f"Number of sampled users: {len(sample_user_ids)}")
        #print(f"Number of ratings for sampled users: {len(gen_file)}")

        return gen_file, sample_user_ids

    @staticmethod
    def find_max_min_users_rated(self):
        pass
    
    @staticmethod
    def find_max_min_movies_rated(self, df_ratings, sampled_userIds=None):
        """
        Find user with max/min number of movies rated.
        userIds: optional subset of users to consider.
        Returns: (userId_max, userId_min)
        """
        if sampled_userIds is not None and len(sampled_userIds) > 0:
            df_ratings_subset = df_ratings[df_ratings['userId'].isin(sampled_userIds)]
        else:
            df_ratings_subset = df_ratings
        
        user_ratings_count = df_ratings_subset.groupby('userId')['rating'].count()
        max_movies = user_ratings_count.max()
        min_movies = user_ratings_count.min()

        user_rated_max_movies = user_ratings_count[user_ratings_count == max_movies].index[0]
        user_rated_min_movies = user_ratings_count[user_ratings_count == min_movies].index[0]

        print(f"Max movies rated by one user: {max_movies}, userId: {user_rated_max_movies}")
        print(f"Min movies rated by one user: {min_movies}, userId: {user_rated_min_movies}")

        return user_rated_max_movies, user_rated_min_movies
    
    @staticmethod
    def split_data_ml20m(gen_file, split_mode='random', test_ratio=0.2, random_seed=42):
        """
        Split dataset per user (20% ratings into test, rest into train).

        Args:
        - gen_file: DataFrame with at least ['userId', 'movieId', 'rating', 'timestamp']
        - split_mode: 'random' or 'seq-aware', if 'seq-aware', then split by timestamp
        - test_ratio: percentage of each user's ratings to put in test set, can considered this the same as train/test ratio
        - random_seed: Use in random mode

        Returns:
        - rating_train: DataFrame
        - rating_test: DataFrame
        """
        if not isinstance(gen_file, pd.DataFrame):
            raise ValueError("gen_file must be a pandas DataFrame")
        
        np.random.seed(random_seed)
        train_rows = []
        test_rows = []

        for user_id, user_data in gen_file.groupby('userId'):
            if split_mode == 'seq-aware':
                # Sort by timestamp (ascending)
                user_data = user_data.sort_values('timestamp')
            else:  # random mode
                # Shuffle the rows
                user_data = user_data.sample(frac=1, random_state=random_seed)

            n_ratings = len(user_data)
            n_test = max(1, int(n_ratings * test_ratio))  # at least one rating in test

            test_rows.append(user_data.tail(n_test))
            train_rows.append(user_data.head(n_ratings - n_test))

        rating_train = pd.concat(train_rows).reset_index(drop=True)
        rating_test = pd.concat(test_rows).reset_index(drop=True)
        return rating_train, rating_test