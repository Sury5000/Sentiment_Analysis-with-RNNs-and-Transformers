# Sentiment Analysis with RNNs and Transformers (IMDB)

## Overview

This project implements an end-to-end sentiment analysis pipeline on the IMDB dataset, progressing from classical deep learning models to advanced Transformer-based architectures.

The focus is on understanding how model performance improves through:

* Better sequence handling
* Architectural enhancements
* Transfer learning
* Use of pretrained Transformer models

---

## Problem Statement

Classify movie reviews into:

* Positive (1)
* Negative (0)

---

## Dataset

* IMDB Movie Reviews Dataset
* 50,000 labeled reviews
* Balanced classes

### Data Split

* Train: 80%
* Validation: 20%
* Test: Standard IMDB test split

---

## Data Pipeline

### Preprocessing

* Lowercasing text
* Tokenization using BERT tokenizer (`bert-base-uncased`)
* Padding and truncation (max length = 200)

### Custom Collate Function

* Converts raw text into tensors:

  * input_ids
  * attention_mask
* Converts labels to float tensors

### DataLoader

* Batch size: 256
* Separate loaders for training, validation, and testing

---

## Model Architectures

### 1. Vanilla GRU

* Embedding layer trained from scratch
* Multi-layer GRU
* Linear output layer

Serves as a baseline model.

---

### 2. Packed Sequence GRU

* Uses `pack_padded_sequence`
* Avoids computation on padded tokens

Improves efficiency and gradient flow.

---

### 3. Bidirectional GRU

* Bidirectional GRU layer
* Captures both forward and backward context

Improves contextual understanding.

---

### 4. Pretrained BERT Embeddings with GRU

* Uses pretrained BERT embedding weights
* Embeddings are frozen
* GRU learns task-specific representations

Introduces transfer learning.

---

### 5. BERT with GRU Hybrid

* BERT generates contextual embeddings
* GRU processes sequence outputs

Combines Transformer representations with sequence modeling.

---

### 6. Pure BERT Models

#### CLS Token Approach

* Uses the [CLS] token representation for classification

#### Pooler Output Approach

* Uses `pooler_output` from BERT

---

### 7. HuggingFace BERT Classifier

* Uses `BertForSequenceClassification`
* Built-in classification head
* Supports multi-class classification

---

## Training Setup

### Loss Functions

* BCEWithLogitsLoss for binary classification
* CrossEntropyLoss for Transformer classifier

### Optimizer

* NAdam

### Scheduler

* ReduceLROnPlateau

### Metrics

* Accuracy using TorchMetrics

---

## Training Pipeline

Custom training loop includes:

* Forward pass
* Loss computation
* Backpropagation
* Metric tracking
* Learning rate scheduling

---

## HuggingFace Trainer API

* Tokenized dataset using `.map()`
* Trainer handles:

  * Training loop
  * Evaluation
  * Logging
  * Checkpointing

---

## Inference

### Pretrained Pipeline

* Model: `distilbert-base-uncased-finetuned-sst-2-english`
* Used for direct sentiment predictions

### Batch Inference

* Applied on IMDB validation set
* Compared predictions with ground truth

---

## Bias Analysis

The model was tested on sentences like:

* "I am from Iraq"
* "I am from USA"

Observation:

* Model showed bias toward negative sentiment for certain inputs

### Mitigation

* Used model with neutral class:

  * `cardiffnlp/twitter-roberta-base-sentiment-latest`

This reduced biased predictions by allowing neutral outputs.

---

## Additional Experiment

### Natural Language Inference (NLI)

* Model: `distilbert-base-uncased-finetuned-mnli`
* Tasks:

  * Entailment
  * Contradiction
  * Neutral classification

---

## Key Learnings

* Padding negatively impacts RNN performance without proper handling
* Packed sequences improve efficiency and learning
* Bidirectional models capture better context
* Pretrained embeddings significantly boost performance
* Transformer models outperform traditional RNNs
* Pretrained models may carry bias from training data

---

## Tech Stack

* PyTorch
* TorchMetrics
* HuggingFace Transformers
* HuggingFace Datasets

---

## Conclusion

This project demonstrates a complete progression in NLP modeling:

* From basic RNN architectures
* To optimized sequence handling
* To transfer learning
* To full Transformer-based systems

It highlights both performance improvements and real-world challenges such as bias in machine learning models.
