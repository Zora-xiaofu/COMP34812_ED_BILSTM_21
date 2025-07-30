# Evidence Detection 

## Task Overview
Given a claim and a piece of evidence, determine if the evidence is relevant to the claim.
You will be provided with more than 21K claim-evidence pairs as training data, and almost
6K pairs as validation data.



## Solution Type - chose B & C

- BERT Embeddings + BiLSTM (Solution B)
We use:
- Used BERT word embeddings as input to a bidirectional LSTM
- Followed by a dropout and linear layer for classification
- Tuned hidden size, dropout, num_lstm_layers, epochs, learning rate

- Fine-tuned BERT Transformer (Solution C)
We use:
- Fine-tuned bert-base-uncased
- Tuned learning rate, batch size, dropout, epochs
- Used AdamW optimizer and cross-entropy loss



## Code Overview & Structure (Solution B)

- "nlu_bilstm.ipynb" : Training script and grid search, save the best model and the best parameter combination
- "bilstm_eval.ipynb" : Evaluation script on dev.csv
- "bilstm_predict.ipynb" : Inference script on test.csv
- "Model_Card_B.md" : Model card of solution B - BERT Embeddings + BiLSTM
- "README_B.md" : This README, explains the Solution B for ED task
- "lstm_models/" : this folder Contains:
    - best model - "best_model_bilstm.pt"
    - all parameter combination results of training model - "grid_search_results_bilstm.json"
    - best parameter combination json file - "best_params_bilstm.json"
    - prediction output file - "Group_21_B.csv"

## How to Run (Solution B)

### 1. Training model
Open "nlu_bilstm.ipynb" in Google Colab. and it will take about 5 hours to run. 
After completion, the following files will be saved in "lstm_models/":
- "best_model_bilstm.pt"
- "best_params_bilstm.json"
- "grid_search_results_bilstm.json"

### 2. Evaluate on dev.csv
Open "bilstm_eval.ipynb" in Google Colab
Make sure "dev-nonlabel.csv", "dev.csv", "best_model_bilstm.pt" and "best_params_bilstm.json" are present.
After running, you will see the Evaluation Results of best model including:
- Accuracy
- F1 Score
- Precision
- Recall

### 3. Predict on test.csv - Inference Demo Code
Open "bilstm_predict.ipynb"
Make sure "test.csv", "best_model_bilstm.pt" and "best_params_bilstm.json" are present.
Prediction file will be saved as:
- "Group_21_B.csv"

## Best model is saved at: (Solution B)

Name: best_model_bilstm.pt
Size: 103.5MB
Location: Google Drive
https://drive.google.com/drive/folders/1_sRwGB1Y8Fj5hnrxjj3pNct9UJtJNSBI?usp=sharing



## Code Overview & Structure (Solution C)

- "nlu_bert.ipynb": Training script and grid search, save the best model and the best parameter combination
- "bert_eval.ipynb": Evaluation script on dev.csv  
- "bert_predict.ipynb": Inference script on test.csv, will export prediction file
- "Model_Card_C.md": Model card of solution C - fine-tuned BERT  
- "README_C.md": This README, explains the Solution C for ED task
- "bert_model/": This folder contains:
    - best model - "best_model_bert.pt" 
    - all parameter combination results of training model - "grid_search_results_bert.json"  
    - best parameter combination json file - "best_params_bert.json"
    - Prediction output file - "Group_21_C.csv"

## How to Run (Solution C)

### 1. Train the Model
Open "nlu_bert.ipynb" in Google Colab
Training will take approximately 5.5 hours.
After training, the following files will be saved in the "bert_model/" folder:
- "best_model_bert.pt"
- "best_params_bert.json"
- "grid_search_results_bert.json"

### 2. Evaluate on dev.csv
Open "bert_eval.ipynb" in Google Colab  
Make sure the following files are present:
- "dev-nonlabel.csv": the dev.csv file without 'label' column
- "dev.csv"
- "best_model_bert.pt"
- "best_params_bert.json"

After running, you will see the Evaluation Results of best model including:
- Accuracy
- F1 Score
- Precision
- Recall

### 3. Predict on test.csv - Inference Demo Code
Open "bert_predict.ipynb" in Google Colab 
Make sure "test.csv", "best_model_bert.pt" and "best_params_bert.json" are available.  
Prediction file will be saved as:  
- "Group_21_C.csv"

## Best model is saved at: (Solution C)

Name: best_model_bert.pt
Size: 417.7MB
Location: Google Drive
https://drive.google.com/drive/folders/1V24i9Ll7UF12i4agk8tCOUZdPVhi3s8O?usp=sharing



## Attribution
- Some part of the code base were adapted from coursework completed in last semester.
- **Repository:** https://huggingface.co/google-bert/bert-base-uncased
- **Paper or documentation:** https://aclanthology.org/N19-1423.pdf

## Use of Generative AI Tools

This project used ChatGPT 4 to assist with:
- Writing and polishing the README file and model card file
- Clarifying function explanations
- Used AI to generate the table of description for hyperparameters used for poster.