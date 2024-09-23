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
        
        self.data = pd.read_csv(file_path)
        self.transform = transform
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.train_set = None
        self.test_set = None
        
        self.train = train
        self.include_sensitive_attr = include_sensitive_attr

        # 범주형 열을 자동으로 수치형으로 변환
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            self.data[column], _ = pd.factorize(self.data[column])

        if random_state:
            self.data = self.data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            
        # Extract sensitive attribute
        if self.include_sensitive_attr:
            sensitive_attr = self.data['gender'].values  # 전체 데이터에 대한 민감한 속성
        else:
            sensitive_attr = None    
        
        # Splitting the dataset into train and test in a semi-supervised fashion
        total_len = len(self.data)

        self.train_data = self.data[:int(0.8 * total_len)].reset_index(drop=True)  # 80% for training
        self.test_data = self.data[int(0.8 * total_len):].reset_index(drop=True)  # 20% for testing

        if self.include_sensitive_attr:
            self.sensitive_attr_train = self.train_data['gender'].values
            self.sensitive_attr_test = self.test_data['gender'].values
        else:
            self.sensitive_attr_train = None
            self.sensitive_attr_test = None
        
        X_train = self.train_data.drop(['target', 'gender'], axis = 1).values.astype(float)
        self.y_train = self.train_data['target'].values

        X_test = self.test_data.drop(['target', 'gender'], axis = 1).values.astype(float)
        self.y_test = self.test_data['target'].values
        
        if transform == "standard":    
            # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
            scaler = StandardScaler().fit(X_train)
            self.X_train_scaled = scaler.transform(X_train)
            self.X_test_scaled = scaler.transform(X_test)
            
        elif transform == "minmax":   
            # Scale to range [0,1]
            minmax_scaler = MinMaxScaler().fit(X_train)
            self.X_train_scaled = minmax_scaler.transform(X_train)
            self.X_test_scaled = minmax_scaler.transform(X_test)

        # Train/test sets 생성 
        # self.train_set = np.concatenate([self.X_train_scaled, np.array(self.y_train).reshape(-1,1)], axis = 1)
        # self.test_set = np.concatenate([self.X_test_scaled, np.array(self.y_test).reshape(-1,1)], axis = 1)
       
        # Convert to torch tensors
        if self.train:
            self.data = torch.tensor(self.X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(self.y_train, dtype=torch.int64)
            if self.include_sensitive_attr:
                self.sensitive_attr = torch.tensor(self.sensitive_attr_train, dtype=torch.int64)
            else:
                self.sensitive_attr = None
        else:
            self.data = torch.tensor(self.X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(self.y_test, dtype=torch.int64)
            if self.include_sensitive_attr:
                self.sensitive_attr = torch.tensor(self.sensitive_attr_test, dtype=torch.int64)
            else:
                self.sensitive_attr = None

        self.semi_targets = torch.zeros_like(self.targets)
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        
        if self.include_sensitive_attr and self.sensitive_attr is not None:
            sensitive_attr_value = int(self.sensitive_attr[index])
            return sample, target, semi_target, index, sensitive_attr_value
        else:
            return sample, target, semi_target, index
    
    
class CustomCSVADDataset(BaseADDataset):
    def __init__(self, root: str, dataset_name: str, n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None):
        super().__init__(root)
        
        self.known_outlier_classes = tuple(range(n_known_outlier_classes))  # n_known_outlier_classes를 기반으로 tuple 생성
       
        file_path = f"{root}/{dataset_name}.csv"
        
        # 사전 훈련용 데이터셋 생성 (민감한 속성 포함하지 않음)
        self.pretrain_dataset = CustomCSVDataset(file_path=file_path, train=True, random_state=random_state, include_sensitive_attr=False)
        
        # 훈련용 데이터셋 생성 (민감한 속성 포함)
        dataset = CustomCSVDataset(file_path=file_path, train=True, random_state=random_state, include_sensitive_attr=True)

        train_set, test_set, y_train, y_test =  dataset.train_set, dataset.test_set, dataset.y_train, dataset.y_test
        
        y_train = pd.Series(y_train)
                
        # Debug: Check unique target values
        print("Unique target values:", y_train.unique())
        
        # Define normal and outlier classes
        self.n_classes = 2  # 0 for normal, 1 for outlier
        self.normal_classes = tuple([0])  # Normal class는 0으로 정의
        self.outlier_classes = tuple([1])  # Outlier class는 1으로 정의


        # **클래스 존재 여부 확인**
        # 데이터셋에 normal_class 및 outlier_class가 있는지 확인
        normal_exists = self.normal_classes[0] in y_train.unique()
        outlier_exists = any([cls in y_train.unique() for cls in self.outlier_classes])
        
        print(f"Normal class exists in dataset: {normal_exists}")
        print(f"Outlier class exists in dataset: {outlier_exists}")

        # 만약 정상 클래스나 이상 클래스가 없으면 에러 발생
        if not normal_exists:
            raise ValueError(f"Normal class {self.normal_classes[0]} not found in the dataset.")
        if not outlier_exists:
            raise ValueError(f"Outlier classes {self.outlier_classes} not found in the dataset.")
                    
        # Create semi-supervised setting
        idx, _, semi_targets, sensitive_attr_list  = create_semisupervised_setting(
            y_train.values,  # target column의 값을 이용
            dataset.sensitive_attr_train,
            self.normal_classes, 
            self.outlier_classes, 
            self.known_outlier_classes, 
            ratio_known_normal, 
            ratio_known_outlier, 
            ratio_pollution
        )
        


        
        dataset.semi_targets[idx] = torch.tensor(semi_targets)
        # Update sensitive attributes in the dataset at positions idx
        if dataset.include_sensitive_attr and dataset.sensitive_attr is not None:
            dataset.sensitive_attr[idx] = torch.tensor(sensitive_attr_list)
        
        # Subset train_set to semi-supervised setup
        self.train_set = Subset(dataset, idx)
        self.test_set = CustomCSVDataset(file_path=file_path, train=False, random_state=random_state)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        
        return train_loader, test_loader

