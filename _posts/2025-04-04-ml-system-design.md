---
title:  "Machine Learning System Design Guide"
author: "Irpan Abdurahman"
mathjax: false
layout: post
categories: media
---

This guide is intended for people preparing for ML system design interviews. It covers key concepts across common domains like regression, classification, NLP, computer vision, and recommendation/ranking/search systems. It also provides a step-by-step framework one can follow to come up with a end-to-end solution. 



# Table of Contents
- [Machine Learning Concepts](#machine-learning-concepts)
  - [Regression](#regression)
  - [Classification](#classification)
  - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
  - [Computer Vision](#computer-vision)
  - [Recommendation & Ranking](#recommendation-and-ranking)
  - [Retrieval](#retrieval)
- [Machine Learning System Design Framework](#machine-learning-system-design-framework)

# 🤖 Machine Learning Concepts <a id="machine-learning-concepts"></a>

## 📈 Regression <a id="regression"></a>
**Goal**: Predict a continuous scalar value.

### 🧪 Offline Metrics
- **Mean Squared Error (MSE)**: Penalizes large errors more.
- **Mean Absolute Error (MAE)**: All errors contribute equally.
- **R-squared (R²)**: Measures how well the model explains the variance.

### 🛠️ Typical Models
- **Linear Regression**  
  - **Use when**: Input and output has linear relationships.  
  - **Tradeoffs**: Simple and interpretable, but underperforms with non-linear patterns.

- **Decision Trees**  
  - **Use when**: You need to model complex, non-linear interactions.  
  - **Tradeoffs**: Prone to overfitting, so it’s better to use pruning or ensemble methods (like Random Forest); interpretable.

- **Random Forest**  
  - **Use when**: Accuracy is critical and data has complex patterns.  
  - **Tradeoffs**: Reduces overfitting compared to DT. Less interpretable, more computationally intensive.

- **Gradient Boosting (e.g., XGBoost, LightGBM)**  
  - **Use when**: You want state-of-the-art accuracy on tabular data.  
  - **Tradeoffs**: Can overfit, sensitive to hyperparameters.

---

## 🏷️ Classification <a id="classification"></a>
**Goal**: Predict a discrete label/category.

### 🧪 Offline Metrics
- **Accuracy**: Overall correct predictions.
- **Precision**: Correct positive predictions / Total predicted positives. (How much positive predicted are actually positive?)
- **Recall**: Correct positive predictions / Total actual positives. (How much actual positive did we predict?)
- **F1 Score**: Harmonic mean of precision and recall.
- **AUC**: Area under the curve. The model’s ability to distinguish between classes at different thresholds.
    - Used to set a threshold for decision. Ex: To determine best K for if Prediction value > K then model predict Positive. 
    - AUC = 1.0 -> perfect model. AUC = 0.5 -> random guess. AUC < 0.5 -> something is wrong.
    - **ROC AUC** vs. **Precision-Recall AUC**: use ROC when classes are balanced, P-R when imbalanced. 

### 🛠️ Typical Models
- **Logistic Regression**  
  - **Use when**: Simple baseline for binary classification.  
  - **Tradeoffs**: Fast and interpretable, may underperform on complex data.

- **Decision Trees**: Splits the data based on feature values to classify data.
    - **Use when**: When the data has non-linear patterns.
    - **Tradeoffs**: Prone to overfitting, but easily interpretable.

- **Random Forest / Gradient Boosted Trees**  
  - **Use when**: Tabular data with non-linear relationships.  
  - **Tradeoffs**: May need tuning, less interpretable, but better genelization.

- **K-Nearest Neighbors (KNN)**:
    - **Use when**: Simple, works well when decision boundaries are complex.
    - **Tradeoffs**: Computationally expensive, especially with large datasets.

- **Support Vector Machines (SVM)**  
  - **Use when**: High-dimensional data and small datasets.  
  - **Tradeoffs**: Can be slow on large datasets, choosing the right kernel can be tricky.

- **Neural Networks**  
  - **Use when**: Large datasets, image/text/audio features.  
  - **Tradeoffs**: Requires more data, less interpretable.

---

## 🗣️ Natural Language Processing <a id="natural-language-processing-nlp"></a>
**Goal**: Analyze or generate human language data.

### 📝 Typical Tasks
- Text Classification: Spam detection, sentiment analysis.
- Named Entity Recognition (NER): Detect names, orgs, places.
- Question Answering: Extractive or generative.
- Summarization: Abstractive or extractive.
- Translation, Language Modeling, Text Generation.
- Retrieval, Text Similarity.
- Speech Tasks: ASR, TTS.

### 📊 Typical Data
- Raw Text: News articles, reviews, transcripts, web crawl.
- Structured Datasets: CSVs with text + labels (e.g., IMDB, SST-2).
- Paired Data: Source-target pairs (e.g., translation, Q&A).
- Multilingual Data: For translation, cross-lingual models.
- Documents: PDFs, HTML, JSON, XML.
- Audio (optional): Paired speech-text for ASR.

### 🧪 Offline Metrics
- **BLEU / ROUGE / METEOR**: Text similarity for translation/summarization.
- **Accuracy / Precision / Recall/ F1**: For classification tasks (e.g., sentiment analysis).
- **Perplexity**: For evaluating language models.

### 🛠️ Typical Models
- **TF-IDF + Logistic Regression / SVM**  
  - **Use when**: Simple text classification or baseline.  
  - **Tradeoffs**: Fast, but lacks semantic understanding.

- **RNN / LSTM / GRU**  
  - **Use when**: Sequential tasks like sentiment analysis or translation.  
  - **Tradeoffs**: Struggle with long-term dependencies.

- **Transformer-based models (e.g., BERT, GPT)**  
  - **Use when**: Most modern NLP tasks.  
  - **Tradeoffs**: Require significant compute and fine-tuning.

---

## 🖼️ Computer Vision  <a id="computer-vision"></a>
**Goal**: Extract insights from image or video data.

### 🧪 Offline Metrics
- **Accuracy**, **Precision/Recall/F1**: For classification.
- **Intersection over Union (IoU)**: For object detection.
- **PSNR / SSIM**: For image reconstruction or enhancement.

### 📝 Typical Tasks
- Classification, Object Detection, Segmentation (Semantic/Instance).
- Image Generation, Captioning, VQA.
- OCR, Pose Estimation, Action Recognition.

### 📊 Typical Data
- Image formats (JPG, PNG), annotated with labels, boxes, or masks.
- Multimodal pairs: image + caption, VQA.
- Video frames.

### 🛠️ Typical Models
- **CNN (Convolutional Neural Networks)**  
  - **Use when**: Image classification or object detection.  
  - **Tradeoffs**: Efficient and accurate; limited context awareness.

- **ResNet, EfficientNet**  
  - **Use when**: Deeper or more efficient image models.  
  - **Tradeoffs**: Better accuracy with optimized performance.

- **YOLO / Faster R-CNN**  
  - **Use when**: Real-time object detection.  
  - **Tradeoffs**: Speed vs. accuracy tradeoff depending on model.

- **ViT (Vision Transformers)**  
  - **Use when**: High-end image recognition with sufficient data.  
  - **Tradeoffs**: Require large datasets and compute.

---

## 🎯 Recommendation & Ranking <a id="recommendation-and-ranking"></a>

**Goal**: Suggest relevant items to users by ranking items based on similarity or a scoring function.

### 📊 Typical Data
- **User Attributes**: Metadata about users (e.g., Demographics, preferences, browsing history).
- **Item Attributes**: Metadata about items (e.g., product descriptions, genres, categories). Labelsindicating relevance or ranking order, or relevance scores.
- **User-Item Interactions**: (User, Item, Score) triplets or implicit feedback such as clicks, views, purchases.  

### 🧪 Offline Metrics
- **Precision@K**: Fraction of recommended items in the top-K list that are relevant.  
  *Example*: If the system recommends 5 items and 3 are relevant, Precision@5 = 60%.

- **Recall@K**: Fraction of relevant items that are successfully recommended in the top-K list.  
  *Example*: If the user has 10 favorite items, and 3 appear in the top-5 recommendations, Recall@5 = 30%.

- **MAP (Mean Average Precision)**: Averaged precision across queries.  
  *Best when multiple relevant items exist, and their position matters.*

- **NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality by assigning higher importance to relevant items ranked higher.  
  *Best when higher rank matters more.*

- **Coverage**: Measures the percentage of items recommended at least once.  
  *To ensure every product has a chance to be recommended.*

- **Diversity**: Ensures that recommended items are varied.  
  *To prevent repetitive recommendations and improve user experience.*
  
### 🛠️ Typical Recommandation Models
- **Collaborative Filtering (Matrix Factorization)**  
  - **Use when**: You have rich user-item interaction data. Recommends items based on past user behaviors.
  - **Tradeoffs**: Cold-start problem (new user no data), ignores metadata.

- **Content-based Filtering**  
  - **Use when**: You lack collaborative data or want interpretability. Recommends items based on the attributes of the items or users. 
  - **Tradeoffs**: Can't capture user-item interactions well.

- **Hybrid Models**  
  - **Use when**: You want to combine strengths of collaborative and content methods.  
  - **Tradeoffs**: Increased complexity.

- **Deep Learning-based (e.g., Two-tower models)**  
  - **Use when**: Rich features and large-scale data.  
  - **Tradeoffs**: More complex and resource-intensive.

### 🛠️ Typical Ranking Models
- **Learning-to-Rank** (e.g., RankNet, LambdaMART)  
  - **Use when**: You need to predict the best order of items.  
  - **Tradeoffs**: Supervised learning requires labeled data and can be complex.

- **Regression-Based Ranking**  
  - **Use when**: You have continuous data and need to rank items based on predicted scores.  
  - **Tradeoffs**: May not capture relationships between items as well as more complex models.

- **Pairwise Ranking**  
  - **Use when**: You want to compare pairs of items to determine the better one.  
  - **Tradeoffs**: Computationally expensive.
---

## 🔍 Retrieval <a id="retrieval"></a>

**Goal**: Find relevant candidates from a large collection.

### 🧪 Offline Metrics
- **Hit Rate (HR)**: Measures how often relevant items appear in the top-K retrieval.  
  *Example*: If a user’s preferred item appears in the top-10 search results, it counts as a hit.

### 🛠️ Typical Models
- **TF-IDF (Term Frequency-Inverse Document Frequency)**  
  - **Use when**: Document retrieval tasks based on keyword relevance.  
  - **Tradeoffs**: Simple but doesn’t capture semantic meaning.

- **BM25** (Best Match 25)  
  - **Use when**: Information retrieval tasks, especially in large text corpora.  
  - **Tradeoffs**: More advanced than TF-IDF, but still simple.

- **Neural Networks (Deep Embeddings)**  
  - **Use when**: Semantic understanding of the data is important.  
  - **Tradeoffs**: Requires large datasets and computational resources.

- **Approximate Nearest Neighbors (ANN)**  
  - **Use when**: Fast, scalable retrieval in large datasets.  
  - **Tradeoffs**: May not always find exact neighbors, but balances speed and accuracy well.


# Machine Learning System Design Framework