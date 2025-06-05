import matplotlib.pyplot as plt 
import numpy as np 
import random
import pandas as pd
from tqdm import tqdm

class UserProfileTools:
    def __init__(self, user_profile):
        self.user_profile = user_profile
        self.profile_userIds = list(user_profile.keys())
        #self.nan_users = self.user_profile.columns[self.user_profile.isnull().any()].tolist()
        
    def print_nan_users(self):
        nan_users = self.user_profile.columns[self.user_profile.isnull().any()].tolist()
        return nan_users
    
    def plot_tag_distribution(self, method='boxplot', user_profile = None):
        """
        Plot tag distribution across 10 user groups.

        Args:
          method: 'boxplot' or 'mean' or 'zero'
                  'zero' = plot distribution of users has zero tags
        """
        Id_num_tags = {userId : len(self.user_profile[userId]['TopTagsWithScores']) if not isinstance(self.user_profile[userId]['TopTagsWithScores'], float) else 0 for userId in self.profile_userIds}

        group_sizes = [1384] * 9 + [user_profile.shape[1] - 1384 * 9]

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

    def print_profile(self):
        return self.cleaned_user_profile

    def print_a_profile(self):
        userId = random.choice(self.userIds)
        print(f"userId: {userId}")
        print(self.cleaned_user_profile.loc[userId, 'ProfileParagraph'])

    def plot_tag_distribution(self, method='boxplot'):
        return super().plot_tag_distribution(method=method)