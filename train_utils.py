import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold

class TrainUtils:
    
    @staticmethod
    def get_skf_split(full_df, num_splits, split_idx, random_state=3141592653): 
        """
        This assumes that there is a single label, and it's the last column
        StratifiedKFold is like KFold, but ensures equal distribution of each class labels
        """
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
        
        # First arg to skf.split() is ignored, output is indices into X/y, not the actual values
        x_dummy = [0] * full_df.shape[0]  
        all_splits = skf.split(x_dummy, full_df['label'].values)
        
        train_indices, test_indices = list(all_splits)[split_idx]

        X_train = full_df.iloc[train_indices, :-1]
        y_train = full_df.iloc[train_indices, -1]
        X_test = full_df.iloc[test_indices, :-1]
        y_test = full_df.iloc[test_indices, -1]
        return X_train.values, y_train.values, X_test.values, y_test.values

    @staticmethod
    def kfold_train_predict(model,
                            full_df,
                            num_splits=5,
                            random_state=3141592653): 
        """
        We expect `model` to be an sklearn model with .fit() and .predict()
        
        This returns (y_true_all, y_pred_all) which can be used for confusion matrix or metrics
        """
        y_true_agg, y_pred_agg = [], []
        for i in range(num_splits):
            X_train, y_train, X_test, y_test = TrainUtils.get_skf_split(
                full_df,
                num_splits,
                i,
                random_state)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_true_agg.append(y_test)
            y_pred_agg.append(y_pred)

        return (np.concatenate(y_true_agg),
                np.concatenate(y_pred_agg))
        
        
        