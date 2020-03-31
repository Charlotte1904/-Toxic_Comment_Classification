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
    over_sampling_group_list = ["combo_6_toxic", "combo_2_obscene_and_insult", "combo_2_hate_and_insult", "combo_3_obscene_and_insult", "combo_2_toxic_and_severe_toxic", "combo_2_toxic_and_obscene", "combo_2_toxic_and_threat", "combo_2_toxic_and_insult", "combo_2_toxic_and_identity_hate", "combo_4_wo_threat"]
    train_indices, val_indices, test_indices, total_index_list = [], [], [], []
    for key_name, index_list in labels_idx_dict.items():
        print(f"{key_name.ljust(40,' ' )} {len(index_list)}")
        total_index_list.extend(index_list)
        ## Calculate size of dataset
        data_size = len(index_list)
        train_size, val_size, test_size = get_train_val_test_sizes(data_size)
        
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

    assert(len(df) == len(train_df)+len(val_df)+len(test_df))

    return train_df, val_df, test_df

if __name__ == "__main__":
    data_csv_path = "data/transformed_train.csv"
    train_csv_path = "data/transformed_train_set.csv"
    val_csv_path = "data/transformed_val_set.csv"
    test_csv_path = "data/transformed_test_set.csv"

    train = pd.read_csv(data_csv_path)

    train_df, val_df, test_df = generate_stratified_train_val_test(train)

    train_df.to_csv(train_csv_path)
    val_df.to_csv(val_csv_path)
    test_df.to_csv(test_csv_path)

    """
    clean                                    143346
    combo_6_toxic                            31
    toxic                                    5666
    combo_2_obscene_and_insult               181
    combo_2_hate_and_insult                  28
    combo_3_obscene_and_insult               3820
    combo_2_toxic_and_severe_toxic           41
    combo_2_toxic_and_obscene                1758
    combo_2_toxic_and_threat                 113
    combo_2_toxic_and_insult                 1215
    combo_2_toxic_and_identity_hate          136
    combo_4_wo_threat                        1620
    other remaining toxic                    1616

    """

