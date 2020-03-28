import random
import numpy as np
import pandas as pd 
from collections import defaultdict, Counter

def get_train_val_test_sizes(data_size):
    val_size = int(data_size * 0.1)
    test_size = int(data_size * 0.15)
    train_size = data_size - val_size - test_size
    return train_size, val_size, test_size

def generate_stratified_train_val_test(df):
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df["toxic_count"] = df.iloc[:,2:].sum(axis=1)

    ## Create different group of toxics 
    labels_idx_dict = defaultdict(list)
    labels_idx_dict["clean"] = df[df["toxic_count"] == 0].sample(frac=1,random_state=9).index
    labels_idx_dict["combo_6_toxic"] = df[df["toxic_count"] == 6].sample(frac=1,random_state=9).index
    labels_idx_dict["toxic"] = df[(df["toxic_count"] == 1) & (df["toxic"] == 1)].sample(frac=1,random_state=9).index
    labels_idx_dict["combo_2_obscene_and_insult"] = df[(df["toxic_count"] == 2) & (df["obscene"] == 1) & (df["insult"] == 1)].sample(frac=1,random_state=9).index
    labels_idx_dict["combo_2_hate_and_insult"] = df[(df["toxic_count"] == 2) & (df["identity_hate"] == 1) & (df["insult"] == 1)].sample(frac=1,random_state=9).index
    labels_idx_dict["combo_3_obscene_and_insult"] = df[(df["toxic_count"] == 3) & (df["obscene"] == 1) & (df["insult"] == 1)].sample(frac=1,random_state=9).index
    labels_idx_dict["combo_4_wo_threat"] = df[(df["toxic_count"] == 4) & (df["threat"] == 0)].sample(frac=1,random_state=9).index
    for label_name in labels[1:]:
        labels_idx_dict[f"combo_2_toxic_and_{label_name}"] = df[(df["toxic_count"] == 2) & (df[label_name] == 1) & (df["toxic"] == 1)].sample(frac=1,random_state=9).index

    ## 
    train_indices, val_indices, test_indices, total_index_list = [], [], [], []
    for key_name, index_list in labels_idx_dict.items():
        print(f"{key_name.ljust(40,' ' )} {len(index_list)}")
        total_index_list.extend(index_list)
        ## Calculate size of dataset
        data_size = len(index_list)
        train_size, val_size, test_size = get_train_val_test_sizes(data_size)

        train_portion = index_list[:train_size]
        val_portion = index_list[train_size:train_size+val_size]
        test_portion = index_list[-test_size:]

        print(f"{key_name.ljust(40,' ' )} train {len(train_portion) == train_size} {len(train_portion)} {train_size}")
        print(f"{key_name.ljust(40,' ' )} val {len(val_portion) == val_size} {len(val_portion)} {val_size}")
        print(f"{key_name.ljust(40,' ' )} test {len(test_portion) == test_size} {len(test_portion)} {test_size}")

        ## Append each portion 
        train_indices.extend(index_list[:train_size])
        val_indices.extend(index_list[train_size:train_size+val_size])
        test_indices.extend(index_list[-test_size:])

    ## Get the remaining group of data
    other_toxic_indices = df[~df.index.isin(total_index_list)].sample(frac=1,random_state=9).index
    data_size = len(other_toxic_indices)
    print(data_size)

    ## Add the remaining data to the train test val indices list
    train_size, val_size, test_size = get_train_val_test_sizes(data_size)
    train_indices.extend(other_toxic_indices[:train_size])
    val_indices.extend(other_toxic_indices[train_size:train_size+val_size])
    test_indices.extend(other_toxic_indices[-test_size:])

    ## Create the dataset from the indices 
    train_df = df[df.index.isin(train_indices)]
    val_df = df[df.index.isin(val_indices)]
    test_df = df[df.index.isin(test_indices)]
    print(len(train_df),len(val_df),len(test_df))

    return train_df, val_df, test_df

if __name__ == "__main__":
    train = pd.read_csv('data/transformed_train.csv')
    train_df, val_df, test_df = generate_stratified_train_val_test(train)