---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/Zora-xiaofu/COMP34812_ED_BILSTM_21

---

# Model Card for e64848xf-j63850yd-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to determine whether a given piece of evidence is relevant to a given claim.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model uses BERT-based word embeddings and a BiLSTM architecture for binary classification.
A grid search over key hyperparameters (learning rate, batch size, dropout, epoch, LSTM size and layers) was conducted to select the best configuration based on F1-score.

- **Developed by:** Xiao Fu and Yutong Dai
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** BiLSTM with BERT embeddings
- **Finetuned from model [optional]:** N/A (uses BERT word embeddings only)

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/google-bert/bert-base-uncased
- **Paper or documentation:** https://aclanthology.org/N19-1423.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The ED task dataset, consisting of over 21K claim-evidence pairs in train.csv and nearly 6K pairs in dev.csv.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 2e-05
      - batch_size: 16
      - dropout: 0.1
      - epochs: 5
      - lstm_hidden_size: 256
      - num_lstm_layers: 2

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 5 hours
      - duration per training epoch: around 32 seconds
      - model size: 103.5MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 2K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of 66% and an accuracy of 79%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: A100 recommended

### Software


      - Transformers 4.50.3
      - Pytorch 2.6.0+cu124
      - scikit-learn 1.6.1
      - pandas 2.2.2
      - tqdm 4.67.1
      - numpy 2.0.2

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This model only uses the first 128 tokens of each input sequence due to the MAX_LEN restriction.
Longer inputs are truncated. 

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Grid search was used to select the best configuration based on F1-score on the development dataset.
The model only uses BERT word embeddings to reduce computational cost and does not employ transformer architectures.
