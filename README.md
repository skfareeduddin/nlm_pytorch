# Neural Language Model Training (PyTorch)  
### Assignment 2

**Author:** Syed Khaja Fareeduddin  
**Date:** 14 November 2025  

This repository contains the implementation and analysis for Assignment 2 of the Speech-to-Speech Machine Translation evaluation.  
The goal is to build character-level neural language models from scratch using PyTorch and study underfitting, overfitting, best-fit behaviour, and perplexity.  
A Transformer model is also implemented for extra credit.

---

## Repository Structure
```
nlm_pytorch/
├─ dataset/
|  └── Pride_and_Prejudice-Jane_Austen.txt
|  
├─ report/
|  └── NLM_Report_SyedKhajaFareeduddin.pdf
|
├─ src/
│  ├── data_preprocessing.py
│  ├── model_lstm.py
│  ├── model_transformer.py
│  ├── train.py
│  ├── evaluate.py
│  └── utils.py
│
├─ outputs/
│  ├── models/
│  │ ├── lstm_bestfit.pth
│  │ ├── lstm_underfit.pth
│  │ ├── transformer.pth
│  │ └── lstm_overfit.pth (download via Google Drive link given below)
│  └── plots/
│  ├── loss_lstm_underfit.png
│  ├── loss_lstm_overfit.png
│  ├── loss_lstm_bestfit.png
│  └── loss_transformer.png
│
└─ NLM.ipynb
```


---

## Dataset

- File: `Pride_and_Prejudice-Jane_Austen.txt`  
- Treated as a character-level language modeling dataset  
- Vocabulary size: 61 characters  
- Preprocessing done in: `src/data_preprocessing.py`

Processing includes:
- Lowercasing text  
- Character tokenization  
- char2idx and idx2char mappings  
- Sliding-window sequence generation  
- Train/validation split: 90% / 10%  
- Sequence length: 100  
- Batch size: 128  

---

## Models Implemented

### 1. LSTM Language Model
File: `model_lstm.py`

Architecture:
- Embedding dimension: 256  
- Hidden size: 512  
- Layers: 2  
- Dropout: 0.3  
- Final linear classifier outputs vocabulary logits  

Trained in three configurations:
- Underfit  
- Overfit  
- Best Fit  

---

### 2. Transformer Language Model (Extra Credit)
File: `model_transformer.py`

Architecture:
- Embedding: 256  
- 4 attention heads  
- 2 encoder layers  
- Feed-forward size: 512  
- Learnable positional encoding  
- Outputs logits over vocabulary for every timestep  

Performs significantly better than LSTMs on this dataset.

---

## Results Summary

| Model | Training Loss | Validation Loss | Perplexity | Behaviour |
|-------|----------------|------------------|------------|-----------|
| LSTM (Underfit) | 1.23 → 1.14 | 10.92 → 11.17 | 71,217 | Too small to learn |
| LSTM (Overfit) | 0.78 → 0.16 | 14.86 → 19.54 | 309,231,617 | Memorized training data |
| LSTM (Best Fit) | 0.93 → 0.29 | 12.31 → 19.07 | 192,327,043 | Limited generalization |
| Transformer | 0.51 → 0.016 | 0.32 → 0.07 | 1.07 | Best performer |

---

## Loss Plots

All loss curves are available in: `outputs/plots/`


Includes plots for:
- LSTM Underfit  
- LSTM Overfit  
- LSTM Best Fit  
- Transformer  

---

## Model Checkpoints

Saved in: `outputs/models/`

Note: GitHub does not allow uploading files larger than 45 MB.  
The overfit LSTM model can be downloaded from Google Drive:
[Overfit Model](https://drive.google.com/file/d/1zw4rsIHDPXQxH_Yftv6Pe5wvYvvObcXF/view?usp=sharing)

Available in the repository:
- lstm_underfit.pth  
- lstm_bestfit.pth  
- transformer.pth  

---

## Running the Project

### 1. Install Dependencies

```
pip install torch numpy matplotlib
```

### 2. Run the Notebook

Open: `NLM.ipynb`

This notebook:
- Loads dataset  
- Trains all models  
- Saves plots and models  
- Computes perplexity  

---

## Extra Credit Completed

- Transformer LM implemented from scratch  
- Very low perplexity achieved (1.07)  
- Loss plots generated  
- Clear documentation and reproducibility  

---

## Report

A complete PDF report summarizing the methodology and results has been submitted separately as required.

---

## Conclusion

This project demonstrates:
- Implementation of neural language models from scratch  
- Behaviour of LSTM under underfitting, overfitting, and balanced scenarios  
- Use of perplexity for evaluation  
- Superior performance of the Transformer model  
- Understanding of model capacity and generalization  

---
