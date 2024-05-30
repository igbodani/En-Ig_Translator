import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from igbo_text import IgboText


def create_dataset():
    ds = tfds.load('huggingface:igbo_english_machine_translation/ig-en')
    train_df1 = pd.read_csv('/content/english-igbo-bible.csv')
    train_df2 = pd.read_csv('/content/english-igbo-dictionary.csv')

    eng = []
    igbo = []

    for split in ['train', 'validation', 'test']:
        for elem in ds[split]:
            eng.append(elem['translation']['en'].numpy().decode('utf-8'))
            igbo.append(elem['translation']['ig'].numpy().decode('utf-8'))

    combined_ig = pd.concat([pd.Series(igbo), train_df1['ig'],train_df2['ig'] ], axis=0).reset_index(drop=True)
    combined_en = pd.concat([pd.Series(eng), train_df1['en'], train_df2['en']], axis=0).reset_index(drop=True)

    # Create the new DataFrame with the combined columns
    combined_df = pd.DataFrame({
        'ig': combined_ig,
        'en': combined_en
    })

    # Replace empty strings with NaN
    combined_df.replace('', pd.NA, inplace=True)

    # Drop rows with NaN values
    combined_df.dropna(inplace=True)


    igbo_sentences = combined_df['ig'].to_list()
    normalized_sentences = []
    # normalize igbo text
    for elem in igbo_sentences:
        ig_text = IgboText()
        normalized_sentences.append(ig_text.normalize(elem, convert_to_lower=True, remove_abbreviations=True))
        combined_df['normalized-ig'] = normalized_sentences

    return combined_df



def load_data():
    combined_df = pd.read_csv('/content/eng-ig.csv')
    total_count = len(combined_df)
    train_size = int(0.80 * total_count)
    valid_test_size = total_count - train_size
    valid_size = int(0.10 * total_count)
    test_size = valid_test_size - valid_size

    # Shuffle the indices
    indices = np.random.permutation(total_count)

    # Get indices for validation and test sets
    valid_indices = indices[:valid_size]
    test_indices = indices[valid_size:valid_size + test_size]

    # Create validation and test sets
    valid_data = combined_df.iloc[valid_indices]
    test_data = combined_df.iloc[test_indices]

    # Remove the validation and test indices from the original data to get the training set
    train_indices = indices[valid_size + test_size:]
    train_data = combined_df.iloc[train_indices]

    return train_data, valid_data, test_data
