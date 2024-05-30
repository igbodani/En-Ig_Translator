# En-Ig_Translator


## Overview
This project implements an English to Igbo translator using a Transformer model. The implementation is based on the TensorFlow blog's [tutorial](https://www.tensorflow.org/text/tutorials/transformer#run_inference) on machine translation with Transformer models. The main components of this project include data preparation, training, evaluation, deployment, and endpoint predictions.

Requirements 

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [References](#references)

## Requirements
To run this project, you need to install the following libraries:

- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Igbo_text


## Data Preparation
Load dataset from Tensorflow [datasets](https://www.tensorflow.org/datasets/community_catalog/huggingface/igbo_english_machine_translation)
Download the Igbo-Bible CSV file from Kaggle [here](https://www.kaggle.com/datasets/goodyduru/english-igbo-bible)
Download the Igbo-Dictionary CSV file from Kaggle [here](https://www.kaggle.com/datasets/goodyduru/english-igbo-dictionary)


```python
ds = tfds.load('huggingface:igbo_english_machine_translation/ig-en')
train_df1 = pd.read_csv('/path/english-igbo-bible.csv')
train_df2 = pd.read_csv('/path/english-igbo-dictionary.csv')

```

Combine English and Igbo sentences into one data frame

``` python
    # Create the new data frame with the combined columns
    combined_df = pd.DataFrame({
        'ig': combined_ig,
        'en': combined_en
    })
```


Normalize Igbo text
```python
 igbo_sentences = combined_df['ig'].to_list()
    normalized_sentences = []
    # Normalize Igbo text
    for elem in igbo_sentences:
        ig_text = IgboText()
        normalized_sentences.append(ig_text.normalize(elem, convert_to_lower=True, remove_abbreviations=True))

```

Tokenize and Transform Data
```python

text_vec_layer_ig = tf.keras.layers.TextVectorization(output_sequence_length=max_length)
text_vec_layer_en = tf.keras.layers.TextVectorization(output_sequence_length=max_length)


def prepare_batch(en_list, ig_list):
  en_input = tf.constant([f"[START] {s} [END]" for s in en_list]) #input for encoder
  ig_input = tf.constant([f"[START] {s}" for s in ig_list]) # input for decoder

  ig_label = text_vec_layer_ig([f"{s} [END]" for s in ig_list])  # output/target of decoder

  return (en_input, ig_input), ig_label


train_ds = tf.data.Dataset.from_tensor_slices((prepare_batch(train_data['en'].to_list(), train_data['normalized-ig'].to_list())))

# Batch the dataset
batch_size = 64
BUFFER_SIZE = 20000
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

```
Repeat steps for validation and test set 

## Model Architecture

Transformer hyperparameters
```python 
num_layers = 4
d_model = 256
dff = 750
num_heads = 8
dropout_rate = 0.1
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
```

## Training
history = transformer.fit(
    train_ds,
    epochs=100,
    validation_data=valid_ds,
    callbacks=[early_stopping]
)

For more information on data processing, model architecture, and training check [here]()
## Usage
Translate your English text to Igbo [here](https://translatorapp-4px6bmbdbq-uw.a.run.app/ ) 

## References
- [Colab Notebook](https://github.com/igbodani/new-plant-diseases/blob/main/PlantVision.ipynb)
- [Neural machine translation with a Transformer and Keras]([https://www.tensorflow.org/](https://www.tensorflow.org/text/tutorials/transformer#run_inference))

