from torch.utils.data import DataLoader, Subset, Dataset
from base.base_dataset import BaseADDataset
from .preprocessing import create_semisupervised_setting
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CustomCSVDataset(Dataset):
    """Dataset class for loading data from a custom CSV file."""
    def __init__(self, file_path, train=True, transform='minmax', random_state=None, include_sensitive_attr=True):
        # self.data = pd.read_csv(file_path)
        self.data = pd.read_csv(file_path, index_col = 0)
        self.data['id'] = self.data['subject_id'].astype('str') + '_' + self.data['hadm_id'].astype('str')
        self.data = self.data.drop(['subject_id', 'hadm_id'], axis = 1)
        self.data.set_index('id', inplace=True)

        self.transform = transform
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.train_set = None
        self.test_set = None

        self.train = train
        self.include_sensitive_attr = include_sensitive_attr

        # Convert categorical columns to numerical
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            self.data[column], _ = pd.factorize(self.data[column])

        if random_state is not None:
            self.data = self.data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        # Extract sensitive attribute
        if self.include_sensitive_attr:
            if 'gender' not in self.data.columns:
                raise ValueError("Sensitive attribute 'gender' not found in the dataset.")
            sensitive_attr = self.data['gender'].values  # Sensitive attribute
        else:
            sensitive_attr = None

        # Splitting the dataset into train and test
        total_len = len(self.data)

        self.train_data = self.data[:int(0.8 * total_len)].reset_index(drop=True)  # 80% for training
        self.test_data = self.data[int(0.8 * total_len):].reset_index(drop=True)  # 20% for testing

        if self.include_sensitive_attr:
            self.sensitive_attr_train = self.train_data['gender'].values
            self.sensitive_attr_test = self.test_data['gender'].values
        else:
            self.sensitive_attr_train = None
            self.sensitive_attr_test = None

        X_train = self.train_data.drop(['target', 'gender'], axis=1).values.astype(float)
        self.y_train = self.train_data['target'].values

        X_test = self.test_data.drop(['target', 'gender'], axis=1).values.astype(float)
        self.y_test = self.test_data['target'].values

        # Apply transformations
        if transform == "standard":
            scaler = StandardScaler().fit(X_train)
            self.X_train_scaled = scaler.transform(X_train)
            self.X_test_scaled = scaler.transform(X_test)
        elif transform == "minmax":
            minmax_scaler = MinMaxScaler().fit(X_train)
            self.X_train_scaled = minmax_scaler.transform(X_train)
            self.X_test_scaled = minmax_scaler.transform(X_test)
        else:
            raise ValueError(f"Unsupported transform type: {transform}")

        # Convert to torch tensors
        if self.train:
            self.data = torch.tensor(self.X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(self.y_train, dtype=torch.int64)
            if self.include_sensitive_attr:
                self.sensitive_attr = torch.tensor(self.sensitive_attr_train, dtype=torch.int64)
            else:
                self.sensitive_attr = None
            self.semi_targets = torch.full_like(self.targets, -1, dtype=torch.int64)  # Initialize as -1
        else:
            self.data = torch.tensor(self.X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(self.y_test, dtype=torch.int64)
            if self.include_sensitive_attr:
                self.sensitive_attr = torch.tensor(self.sensitive_attr_test, dtype=torch.int64)
            else:
                self.sensitive_attr = None
            self.semi_targets = None  # No semi_targets for test set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, target = self.data[index], int(self.targets[index])
        if self.train:
            semi_target = int(self.semi_targets[index])
            if self.include_sensitive_attr and self.sensitive_attr is not None:
                sensitive_attr_value = int(self.sensitive_attr[index])
                return sample, target, semi_target, index, sensitive_attr_value
            else:
                return sample, target, semi_target, index
        else:
            if self.include_sensitive_attr and self.sensitive_attr is not None:
                sensitive_attr_value = int(self.sensitive_attr[index])
                return sample, target, index, sensitive_attr_value
            else:
                return sample, target, index

class CustomCSVADDataset(BaseADDataset):
    def __init__(self, root: str, dataset_name: str, n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None):
        super().__init__(root)

        self.known_outlier_classes = tuple(range(n_known_outlier_classes))  # Create tuple based on n_known_outlier_classes

        file_path = f"{root}/{dataset_name}.csv"

        # Pretraining dataset without sensitive attributes
        self.pretrain_dataset = CustomCSVDataset(
            file_path=file_path,
            train=True,
            random_state=random_state,
            include_sensitive_attr=False
        )

        # Training dataset with sensitive attributes
        dataset = CustomCSVDataset(
            file_path=file_path,
            train=True,
            random_state=random_state,
            include_sensitive_attr=True
        )

        # Access y_train and y_test correctly
        y_train = dataset.y_train
        y_test = dataset.y_test

        # Convert to pandas Series for processing
        y_train_series = pd.Series(y_train)

        # Define normal and outlier classes
        self.n_classes = 2  # 0 for normal, 1 for outlier
        self.normal_classes = (0,)  # Define normal class as 0
        self.outlier_classes = (1,)  # Define outlier class as 1

        # **Check for class existence**
        normal_exists = self.normal_classes[0] in y_train_series.unique()
        outlier_exists = any([cls in y_train_series.unique() for cls in self.outlier_classes])

        print(f"Normal class exists in dataset: {normal_exists}")
        print(f"Outlier class exists in dataset: {outlier_exists}")

        # Raise error if classes are missing
        if not normal_exists:
            raise ValueError(f"Normal class {self.normal_classes[0]} not found in the dataset.")
        if not outlier_exists:
            raise ValueError(f"Outlier classes {self.outlier_classes} not found in the dataset.")

        # Create semi-supervised setting
        list_idx, list_labels, list_semi_labels, sensitive_attr_list = create_semisupervised_setting(
            y_train_series.values,  # Use target column values
            dataset.sensitive_attr_train,
            self.normal_classes,
            self.outlier_classes,
            self.known_outlier_classes,
            ratio_known_normal,
            ratio_known_outlier,
            ratio_pollution,
            random_state=random_state
        )

        # Assign semi_targets
        # Initialize semi_targets as -1
        semi_targets = dataset.semi_targets.numpy()
        semi_targets[list_idx] = list_semi_labels
        dataset.semi_targets = torch.tensor(semi_targets, dtype=torch.int64)

        # Assign sensitive attributes are already handled in CustomCSVDataset

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(dataset, list_idx)

        # **Ensure include_sensitive_attr=True for test_set**
        self.test_set = CustomCSVDataset(
            file_path=file_path,
            train=False,
            random_state=random_state,
            include_sensitive_attr=True  # Explicitly include sensitive attributes
        )

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=True
        )
        test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers,
            drop_last=False  # Keep drop_last=False for testing to include all samples
        )

        return train_loader, test_loader

