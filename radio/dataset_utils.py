import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Union, Generator, Tuple
import os


def split_dataset(dataset_df: Union[str, pd.DataFrame],
                  train_size: Union[int, float] = 100,
                  valid_size: Union[int, float] = 15,
                  test_size: Union[int, float] = 30,
                  random_state: int = 42,
                  save=False
                 ):
    
    if type(dataset_df) == str:
        dataset_df = pd.read_csv(dataset_df)
        
    train_set, test_set = train_test_split(dataset_df, train_size=train_size, random_state=random_state, stratify=dataset_df["label"])
    
    if train_size > 1:
        valid_set, test_set = train_test_split(test_set, train_size=valid_size, test_size=test_size, random_state=random_state, stratify=test_set["label"])
    else:
        valid_set, test_set = train_test_split(test_set, test_size=2/3, random_state=random_state, stratify=test_set["label"])
    
    if save:
        train_set.to_csv("set_train.csv", index=False)
        valid_set.to_csv("set_validation.csv", index=False)
        test_set.to_csv("set_test.csv", index=False)
    
    return train_set, valid_set, test_set