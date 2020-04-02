import random
import numpy as np
import pandas as pd 
from collections import defaultdict, Counter

def get_train_val_test_sizes(data_size, val_ratio, test_ratio):
    val_size = int(data_size * val_ratio)
    test_size = int(data_size * test_ratio)
    train_size = data_size - val_size - test_size
    return train_size, val_size, test_size

def create_sub_sample(df, index_list, n_duplicates):
    subset_df = df[df.index.isin(index_list)]
    list_of_subsets = [subset_df]*n_duplicates
    # subset_df = combine_subset(pd.DataFrame(), list_of_subsets)
    combined_df = pd.concat(list_of_subsets, axis = 0, ignore_index = False)
    return combined_df

def get_combo_labels_dict(df):
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

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
    return labels_idx_dict

def print_labels_combo_dict(labels_idx_dict):
    for key_name, index_list in labels_idx_dict.items():
        print(f"{key_name.ljust(40,' ' )} {len(index_list)}")

def generate_stratified_train_val_test(df, val_ratio, test_ratio):
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    OVERSAMPLING_TRAIN_MAX = 5000
    OVERSAMPLING_VAL_MAX = 500
    df["toxic_count"] = df.iloc[:,2:].sum(axis=1)

    labels_idx_dict = get_combo_labels_dict(df)
    print(f"\n{' ' * 10 } DATA DISTRIBUTION")

    over_sampling_group_list = ["combo_6_toxic", "combo_2_obscene_and_insult", "combo_2_hate_and_insult", "combo_3_obscene_and_insult", "combo_2_toxic_and_severe_toxic", "combo_2_toxic_and_obscene", "combo_2_toxic_and_threat", "combo_2_toxic_and_insult", "combo_2_toxic_and_identity_hate", "combo_4_wo_threat"]
    train_indices, val_indices, test_indices, total_index_list = [], [], [], []
    train_df_subset_list, val_df_subset_list, test_df_subset_list = [], [], []
    for key_name, index_list in labels_idx_dict.items():
        print(f"{key_name.ljust(40,' ' )} {len(index_list)}")
        total_index_list.extend(index_list)
        ## Calculate size of dataset
        data_size = len(index_list)
        train_size, val_size, test_size = get_train_val_test_sizes(data_size, val_ratio, test_ratio)
        
        ## Get indexes for each set
        train_portion = index_list[:train_size]
        val_portion = index_list[train_size:train_size+val_size]
        test_portion = index_list[-test_size:]

        ## Append each portion 
        train_indices.extend(train_portion)
        val_indices.extend(val_portion)
        test_indices.extend(test_portion)

        ## oversampling data
        if key_name in over_sampling_group_list:
            train_n_duplicates = OVERSAMPLING_TRAIN_MAX // len(train_portion) if len(train_portion) < OVERSAMPLING_TRAIN_MAX else 1
            train_df_subset_list.append(create_sub_sample(df,train_portion,train_n_duplicates))
            val_n_duplicates = OVERSAMPLING_VAL_MAX // len(val_portion) if len(val_portion) < OVERSAMPLING_VAL_MAX else 1
            val_df_subset_list.append(create_sub_sample(df,val_portion,val_n_duplicates))

    ## Get the remaining group of data
    other_toxic_indices = df[~df.index.isin(total_index_list)].sample(frac=1,random_state=9).index
    key_name = "remaing_toxic"
    print(f"{key_name.ljust(40,' ' )} {len(other_toxic_indices)}")

    ## Add the remaining data to the train test val indices list
    data_size = len(other_toxic_indices)
    train_size, val_size, test_size = get_train_val_test_sizes(data_size, val_ratio, test_ratio)
     
    ## Get indexes for each set
    train_portion = other_toxic_indices[:train_size]
    val_portion = other_toxic_indices[train_size:train_size+val_size]
    test_portion = other_toxic_indices[-test_size:]

    ## Append each portion 
    train_indices.extend(train_portion)
    val_indices.extend(val_portion)
    test_indices.extend(test_portion)

    ## Oversampling data
    train_n_duplicates = OVERSAMPLING_TRAIN_MAX // len(train_portion) if len(train_portion) < OVERSAMPLING_TRAIN_MAX else 1
    train_df_subset_list.append(create_sub_sample(df,train_portion,train_n_duplicates))
    val_n_duplicates = OVERSAMPLING_VAL_MAX // len(val_portion) if len(val_portion) < OVERSAMPLING_VAL_MAX else 1
    val_df_subset_list.append(create_sub_sample(df,val_portion,val_n_duplicates))

    ## Create the dataset from the indices 
    train_df = df[df.index.isin(train_indices)]
    val_df = df[df.index.isin(val_indices)]
    test_df = df[df.index.isin(test_indices)]

    assert(len(df) == len(train_df)+len(val_df)+len(test_df))

    ## Add oversampling data
    train_df_subset_list.append(train_df)
    combined_train_df = pd.concat(train_df_subset_list, axis = 0, ignore_index = False)
    val_df_subset_list.append(val_df)
    combined_val_df = pd.concat(val_df_subset_list, axis = 0, ignore_index = False)

    print(f"\n{' ' * 10 } TRAIN DATA DISTRIBUTION")
    labels_idx_dict = get_combo_labels_dict(combined_train_df)
    print_labels_combo_dict(labels_idx_dict)

    print(f"\n{' ' * 10 } DATASET")

    print(f"TRAIN SIZE   {len(combined_train_df)}")
    print(f"VAL SIZE     {len(combined_val_df)}")
    print(f"TEST SIZE    {len(test_df)}")

    return combined_train_df.sample(frac=1,random_state=9), combined_val_df.sample(frac=1,random_state=9), test_df

if __name__ == "__main__":
    ### INPUT
    data_csv_path = "data/transformed_train.csv"
    VAL_RATIO = TEST_RATIO = 0.1

    ## OUTPUT
    train_csv_path = "data/transformed_train_set.csv"
    val_csv_path = "data/transformed_val_set.csv"
    test_csv_path = "data/transformed_test_set.csv"
    
    ## Generate train test split
    train = pd.read_csv(data_csv_path)
    train_df, val_df, test_df = generate_stratified_train_val_test(train,VAL_RATIO,TEST_RATIO)

    ## Export datasets to csv files
    train_df.to_csv(train_csv_path)
    val_df.to_csv(val_csv_path)
    test_df.to_csv(test_csv_path)

    """
            
                DATA DISTRIBUTION
        clean                                    143346
        combo_6_toxic                            31
        toxic                                    5666
        combo_2_obscene_and_insult               181
        combo_2_hate_and_insult                  28
        combo_3_obscene_and_insult               3820
        combo_4_wo_threat                        1620
        combo_2_toxic_and_severe_toxic           41
        combo_2_toxic_and_obscene                1758
        combo_2_toxic_and_threat                 113
        combo_2_toxic_and_insult                 1215
        combo_2_toxic_and_identity_hate          136
        remaing_toxic                            1616

                TRAIN DATA DISTRIBUTION
        clean                                    114678
        combo_6_toxic                            5025
        toxic                                    4534
        combo_2_obscene_and_insult               5075
        combo_2_hate_and_insult                  5016
        combo_3_obscene_and_insult               6112
        combo_4_wo_threat                        5184
        combo_2_toxic_and_severe_toxic           5016
        combo_2_toxic_and_obscene                5632
        combo_2_toxic_and_threat                 5005
        combo_2_toxic_and_insult                 5838
        combo_2_toxic_and_identity_hate          5060

                DATASET
        TRAIN SIZE   177351
        VAL SIZE     21110
        TEST SIZE    15952

    """

