import matplotlib.pyplot as plt 
import numpy as np 
import random
import pandas as pd
from tqdm import tqdm
import json

class UserProfileTools:
    def __init__(self, user_profile):
        self.user_profile = user_profile # can be DataFrame or Dict
        self.true_user_profile = self.user_profile.drop(self.user_profile.index[-1]) if self.user_profile.shape[0] == 6 else self.user_profile
        self.df_user_profile = pd.DataFrame(self.true_user_profile)

        self.profile_userIds = list(user_profile.keys()) if isinstance(user_profile, dict) else self.user_profile.index.tolist()
        self.user_serializable = {int(k): v for k, v in user_profile.items()}

    def print_nan_users(self):
        nan_users = self.true_user_profile.columns[self.true_user_profile.isnull().any()].tolist()
        return nan_users
    
    def plot_tag_distribution(self, method='boxplot'):
        """
        Plot tag distribution across 10 user groups.

        Args:
          method: 'boxplot' or 'mean' or 'zero'
                  'zero' = plot distribution of users has zero tags
        """
        Id_num_tags = {userId : len(self.df_user_profile[userId]['TopTagsWithScores']) if not isinstance(self.df_user_profile[userId]['TopTagsWithScores'], float) else 0 for userId in self.profile_userIds}
        
        group_sizes = [1384] * 9 + [self.df_user_profile.shape[1] - 1384 * 9]

        # each user_groups[i] will contain list[number of tags] * group_sizes[i]
        user_groups = []
        start = 0
        for size in group_sizes:
            keys_in_group = list(Id_num_tags.keys())[start : start + size]
            group = [Id_num_tags[key] for key in keys_in_group]
            user_groups.append(group)
            start += size

        # boxplot
        if method == 'boxplot':
            tag_distributions = [[num_tags for num_tags in group] for group in user_groups]

            plt.figure(figsize=(12, 6))
            plt.boxplot(tag_distributions, showfliers=True, patch_artist=True)
            plt.ylabel('Number of Tags')

        elif method != 'boxplot':
            # zeros distribution
            if method == 'zero':
                zero_counts = [len([num_tags for num_tags in group if num_tags == 0]) for group in user_groups]
                counts = zero_counts
            # mean distribution
            elif method == 'mean':
                means = [np.round(np.mean([num_tags for num_tags in group]), decimals = 4) for group in user_groups]
                counts = means
            elif method == 'median':
                medians = [np.round(np.median([num_tags for num_tags in group]), decimals = 4) for group in user_groups]
                counts = medians

            plt.figure(figsize = (12, 6))
            bars = plt.bar(range(1, 11), counts)
            plt.ylabel('Number of Users')
            plt.title('Tag Distribution')

            # insert values on bars
            for i, bar in enumerate(bars):
              height = bar.get_height()
              plt.text(bar.get_x() + bar.get_width()/2., height, str(height), ha='center', va='bottom', fontsize=10)

        else:
            raise ValueError("Invalid method. 'boxplot' or 'mean' or 'zero'.")

        x_labels = [f'Group {i+1}' for i in range(10)]
        plt.xticks(ticks=range(1, 11), labels=x_labels)
        plt.xlabel('User Groups')
        plt.title('Tag Distribution')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def visualize_user_profile(self, print_out = False, return_nan = False):
        nan_users = []
        tag_counts = {}
    
        for user_id, profile in self.user_serializable.items():

            if "TopTagsWithScores" in profile and isinstance(profile["TopTagsWithScores"], list):
                tag_counts[user_id] = len(profile["TopTagsWithScores"])
                print(f"User {user_id}: {tag_counts[user_id]} TopTagsWithScores") if print_out else None
            else:
                print(f"User {user_id} has an error: {profile.get('error')}") if print_out else None # Print the error message if available
                nan_users.append(user_id)

        # Print the tag counts for each user
        print(f"Number of nan users {len(nan_users)}")
        return nan_users if return_nan else None

    def check_user_profile(self, print_seq_503 = False, print_seq_429 = False):
        
        error_503_count = 0
        error_429_count = 0
        errored_users = []

        consecutive_503_count = 0
        max_consecutive_503 = 0
        consecutive_503_sequences = []

        consecutive_429_count = 0 
        max_consecutive_429 = 0 
        consecutive_429_sequences = [] 


        for user_id, profile in self.user_serializable.items():
            if "TopTagsWithScores" in profile and isinstance(profile["TopTagsWithScores"], list):
                # Reset count if successful
                consecutive_503_count = 0 
                consecutive_429_count = 0 
                continue

            else:
                errored_users.append(user_id)
                error_message = profile.get("error")

                if "503 UNAVAILABLE" in error_message:
                    error_503_count += 1
                    consecutive_503_count += 1
                    consecutive_429_count = 0 # Reset 429 count on 503 error
                    max_consecutive_503 = max(max_consecutive_503, consecutive_503_count)

                    # Track consecutive sequences for 503
                    if consecutive_503_count == 1:
                        consecutive_503_sequences.append([user_id])
                    else:
                        consecutive_503_sequences[-1].append(user_id)

                elif "429 RESOURCE_EXHAUSTED" in error_message:
                    error_429_count += 1
                    consecutive_429_count += 1 
                    consecutive_503_count = 0 # Reset 503 count on 429 error
                    max_consecutive_429 = max(max_consecutive_429, consecutive_429_count)

                    # Track consecutive sequences for 429
                    if consecutive_429_count == 1:
                        consecutive_429_sequences.append([user_id])
                    else:
                        consecutive_429_sequences[-1].append(user_id)

                else:
                    consecutive_503_count = 0 # Reset count for 503 for other errors
                    consecutive_429_count = 0 # Reset 429 count for other errors


        print(f"Number of error 503 (UNAVAILABLE): {error_503_count}")
        print(f"Number of error 429 (RESOURCE_EXHAUSTED): {error_429_count}")

        print(f"Max consecutive 503 errors: {max_consecutive_503}")
        if print_seq_503:
            print("All sequence of 503 errors:")
            for seq in consecutive_503_sequences:
                if len(seq) > 1:
                    print(f"  {seq}")

        print(f"Max consecutive 429 errors: {max_consecutive_429}")
        if print_seq_429:
            print("All sequence of 429 errors:") 
            for seq in consecutive_429_sequences:
                if len(seq) > 1: 
                    print(f"  {seq}")

    @staticmethod
    def save_user_profile(final_user_profile, output_):
        # Convert to dict -> save as json file without datatype error
        final_user_json = final_user_profile.to_dict(orient='index')

        output_file = 'completed_user_profile.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_user_json, f, indent=2, ensure_ascii=False)

# Class for cleaning user profiles and converting them to paragraphs
class CleanedUserProfile(UserProfileTools):
    def __init__(self, user_profile):
        super().__init__(user_profile)
        self.userIds = list(user_profile.keys())
        self.cleaned_user_profile = pd.DataFrame(index=self.userIds, columns=['ProfileParagraph'])
        self.cleaned_user_profile = self._profile_to_paragraph()
        self.Id_tags = {userId: len(self.user_profile[userId]['TopTagsWithScores']) if not isinstance(self.user_profile[userId]['TopTagsWithScores'], float) else 0 for userId in self.userIds}

    def __repr__(self):
        return str(self.cleaned_user_profile)

    @staticmethod
    def list_to_string(input_list):
        if isinstance(input_list, float):
            return str(input_list)
        else:
            return ', '.join([str(item) for item in input_list])

    def _clean_decade(self):
        cleaned_decade = {}
        for userId in self.userIds:
            decade = self.user_profile[userId]['FavoriteDecade']
            cleaned_decade[userId] = str(decade)
        return cleaned_decade

    def _clean_genres(self):
        cleaned_genres = {}
        for userId in self.userIds:
            genres = self.user_profile[userId]['FavoriteGenres']
            cleaned_genres[userId] = self.list_to_string(genres)
        return cleaned_genres

    def _clean_actors(self):
        cleaned_actors = {}
        for userId in self.userIds:
            actors = self.user_profile[userId]['FavoriteActors']
            cleaned_actors[userId] = self.list_to_string(actors)
        return cleaned_actors

    def _clean_directors(self):
        cleaned_directors = {}
        for userId in self.userIds:
            directors = self.user_profile[userId]['FavoriteDirectors']
            cleaned_directors[userId] = self.list_to_string(directors)
        return cleaned_directors

    def _clean_tags(self):
        cleaned_tags = {}
        for userId in self.userIds:
            str_tags = ""
            list_tags = self.user_profile[userId]['TopTagsWithScores']
            if isinstance(list_tags, float):
                str_tags = str(list_tags)
            else:
                for item in list_tags:
                    tag = item['tag']
                    score = item['score']
                    str_tags += f"{tag} : {score:.4f}, "
            cleaned_tags[userId] = str_tags[:-2] if str_tags else ""
        return cleaned_tags

    def _profile_to_paragraph(self):
        text_decade = self._clean_decade()
        text_genres = self._clean_genres()
        text_actors = self._clean_actors()
        text_directors = self._clean_directors()
        text_tags = self._clean_tags()
        for userId in tqdm(self.userIds, desc="Creating profile paragraphs", unit="user"):
            paragraph = f"FavoriteDecade: {text_decade[userId]}. FavoriteGenres: {text_genres[userId]}. FavoriteActors: {text_actors[userId]}. FavoriteDirectors: {text_directors[userId]}. TopTagsWithScores: {text_tags[userId]}"
            self.cleaned_user_profile.at[userId, 'ProfileParagraph'] = paragraph
        return self.cleaned_user_profile

    def print_a_profile(self, userId=None):
        """
        Print a single user's profile paragraph. If userId is None, pick a random user.
        """
        if userId is None:
            userId = random.choice(self.userIds)
        print(f"userId: {userId}")
        print(self.cleaned_user_profile.loc[userId, 'ProfileParagraph'])

    def plot_tag_distribution(self, method='boxplot'):
        return super().plot_tag_distribution(method=method)