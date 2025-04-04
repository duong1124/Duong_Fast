import pandas as pd
import numpy as np
from scipy import sparse


class DataLoader:
    """
    Contain methods to read and split dataset into training set and test set.
    All dataset is preprocessed by mapping ID according to the training set and return as `ndarray`.
    (IDs which are not included in the training set remain the same).

    Args:
        data_folder (string): Path to folder that contain dataset
        genome_folder (string): Path to folder that contain the genome_scores file
    """
    def __init__(self, data_folder, genome_folder=None):
        self.__data_folder = data_folder
        if genome_folder is None:
            self.__genome_folder = data_folder
        else:
            self.__genome_folder = genome_folder

        self.__train_data = None
        self.__val_data = None
        self.__test_data = None

    def __read_csv(self, path, columns):
        return pd.read_csv(
            self.__data_folder + "/" + path,
            header=0, names=columns
        )

    def __create_id_mapping(self):
        if self.__val_data:
            unique_uIds = pd.concat([self.__train_data.u_id, self.__test_data.u_id, self.__val_data.u_id]).unique()
            unique_iIds = pd.concat([self.__train_data.i_id, self.__test_data.i_id, self.__val_data.i_id]).unique()
        else:
            unique_uIds = pd.concat([self.__train_data.u_id, self.__test_data.u_id]).unique()
            unique_iIds = pd.concat([self.__train_data.i_id, self.__test_data.i_id]).unique()

        self.user_dict = {uId: idx for idx, uId in enumerate(unique_uIds)}
        self.item_dict = {iId: idx for idx, iId in enumerate(unique_iIds)}

    def __preprocess(self, data):
        """Map the id of all users and items according to user_dict and item_dict.
        To create the user_dict, all user ID in the training set is first sorted, then the first ID is map to 0 and so on.
        Do the same for item_dict.
        This process is done via `self.__create_id_mapping()`.

        Args:
            data (Dataframe): The dataset that need to be preprocessed.

        Returns:
            ndarray: The array with all id mapped.
        """
        # data['u_id'] = data['u_id'].replace(self.user_dict)
        # data['i_id'] = data['i_id'].replace(self.item_dict)

        data['u_id'] = data['u_id'].map(self.user_dict)
        data['i_id'] = data['i_id'].map(self.item_dict)

        # Tag unknown users/items with -1 (when val)
        data.fillna(-1, inplace=True)

        data['u_id'] = data['u_id'].astype(np.int32)
        data['i_id'] = data['i_id'].astype(np.int32)

        return data[['u_id', 'i_id', 'rating']].values

    def load_csv2ndarray(self, train_path="rating_train.csv", test_path="rating_test.csv", val_path="rating_val.csv",  use_val=False, columns=['u_id', 'i_id', 'rating', 'timestamp']):
        """
        Load training set, validate set and test set via `.csv` file.
        Each as `ndarray`.

        Args:
            train_path (string): path to the training set csv file inside self.__data_folder
            test_path (string): path to the testing set csv file inside self.__data_folder
            val_path (string): path to the validating set csv file inside self.__data_folder
            use_val (boolean): Denote if loading validate data or not. Defaults to False.
            columns (list): Columns name for DataFrame. Defaults to ['u_id', 'i_id', 'rating', 'timestamp'].

        Returns:
            train, val, test (np.array): Preprocessed data.
        """
        self.__train_data = self.__read_csv(train_path, columns)
        self.__test_data = self.__read_csv(test_path, columns)

        if use_val:
            self.__val_data = self.__read_csv(val_path, columns)

        self.__create_id_mapping()

        self.__train_data = self.__preprocess(self.__train_data)
        self.__test_data = self.__preprocess(self.__test_data)

        if use_val:
            self.__val_data = self.__preprocess(self.__val_data)
            return self.__train_data, self.__val_data, self.__test_data
        else:
            return self.__train_data, self.__test_data

    def load_genome_fromcsv(self, genome_file="genome_scores.csv", columns=["i_id", "g_id", "score"], reset_index=False):
        """
        Load genome scores from file.
        Args:
            genome_file (string): File name that contain genome scores. Must be in csv format.
            columns (list, optional): Columns name for DataFrame. Must be ["i_id", "g_id", "score"] or ["i_id", "score", "g_id"].
            reset_index (boolean): Reset the genome_tag column or not. Defaults to False.

        Returns:
            scores (DataFrame)
        """
        genome = pd.read_csv(
            self.__genome_folder + "/" + genome_file,
            header=0, names=columns
        )

        if reset_index:
            tag_map = {genome.g_id: newIdx for newIdx, genome in genome.loc[genome.i_id == 1].iterrows()}
            genome["g_id"] = genome["g_id"].map(tag_map)

        genome['i_id'] = genome['i_id'].map(self.item_dict)
        genome.fillna(0, inplace=True)

        return sparse.csr_matrix((genome['score'], (genome['i_id'].astype(int), genome['g_id'].astype(int)))).toarray()
    
    def load_item_features_fromcsv(self, item_features_csv, train_set):
        """
        Map item features from CSV file (indexed by itemId) to item_idx, 
        and filter only items present in the train_set.

        Args:
            item_features_csv: Path to the CSV file containing item features. ['itemId', 'feature's...]
            train_set (np.array)

        Returns:
            item_latent_matrix (np.array): Rows ~ item_idx (from 0 to ...), Cols ~ features.
        """
        item_features = pd.read_csv(item_features_csv, header=None, index_col=0)  # Index is itemId

        # Map itemId to item_idx as a new column
        item_features['item_idx'] = item_features.index.map(self.item_dict)

        train_item_idx = np.unique(train_set[:, 1])

        # Filter item_features to include only items in train_set
        item_features = item_features[item_features['item_idx'].isin(train_item_idx)]

        # Drop rows with itemId not found in self.item_dict
        item_features = item_features.dropna(subset=['item_idx'])
        item_features['item_idx'] = item_features['item_idx'].astype(int)

        # set item_idx as index and sort by item_idx
        item_features = item_features.set_index('item_idx').sort_index()

        item_latent_matrix = item_features.to_numpy()

        return item_latent_matrix

    def calculate_user_latent_matrix(self, item_latent_matrix, train_set):
        """
        Calculate user latent matrix based on the ratings and item latent vector,
        and filter only users present in the train_set.

        Args:
            item_latent_matrix (np.array): Row corresponds to the feature vector of an item (Q_i).
            train_set (np.array)

        Returns:
            user_latent_matrix (np.array): Row corresponds to the profile vector of a user (P_u).
        """
        
        train_user_idx = np.unique(train_set[:, 0])

        # Filter user_dict to include only users in train_set
        train_user_dict = {user_id: user_idx for user_id, user_idx in self.user_dict.items() if user_idx in train_user_idx}

        num_users = len(train_user_dict)
        latent_dim = item_latent_matrix.shape[1]
        user_latent_matrix = np.zeros((num_users, latent_dim))

        for user_id, user_idx in train_user_dict.items():
            # Get all movies rated by the user
            user_ratings = train_set[train_set[:, 0] == user_idx]
            if user_ratings.size == 0:
                continue

            # Extract movie indices and ratings
            movie_indices = user_ratings[:, 1]
            ratings = user_ratings[:, 2]

            # Normalize ratings to [0, 1] - min-max scaling or divided by 5 ?
            normalized_ratings = ratings / 5.0
            # normalized_ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min()) if ratings.max() > ratings.min() else ratings

            # P_u = (1 / |R(u)|) * sum (r_ui * Q_i)
            user_latent_matrix[user_idx] = np.sum(normalized_ratings[:, None] * item_latent_matrix[movie_indices.astype(int)], axis=0) / len(movie_indices)

        return user_latent_matrix
