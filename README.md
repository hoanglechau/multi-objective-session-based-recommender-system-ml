# Large-Scale Multi-Objective Session-Based Recommender System

## 📌 Project Overview

This repository contains the implementation of a high-performance **Two-Stage Recommender System** designed to predict user intent (Clicks, Cart Additions, Orders) from anonymous, short-session e-commerce data.

Developed as a solution for the **OTTO Multi-Objective Recommender Challenge**, this system addresses the "Cold Start" problem in sessions containing over **220 million interaction events**. The solution implements a "Candidate Generation + Ranking" architecture to balance high recall with high precision, achieving a **Recall@20 score of 0.429** on the private leaderboard.

### Key Technical Challenges
* **Extreme Sparsity:** User-item interaction density of only 0.0005%.
* **Data Scale:** Processing 216M+ training events under strict memory constraints.
* **Multi-Objective Optimization:** Simultaneously optimizing for three distinct targets with hierarchical intent levels.

---

## 🏗 System Architecture

The project utilizes a tiered architecture common in large-scale industrial recommender systems (e.g., YouTube, Netflix):

```mermaid
graph TD
    A[Raw Data (JSONL)] -->|Polars Chunking| B(Preprocessing & Denoising)
    B --> C{Stage 1: Candidate Generation}
    C -->|Co-visitation Matrix 1| D[Click-to-Click]
    C -->|Co-visitation Matrix 2| E[Click-to-Cart/Order]
    C -->|Co-visitation Matrix 3| F[Buy-to-Buy]
    D & E & F --> G[Candidate Pool Retrieval]
    G --> H{Stage 2: Ranking}
    H -->|Feature Engineering| I[Contextual & Historical Features]
    I --> J[LightGBM Rankers (x3)]
    J --> K[Final Ensemble & Re-Ranking]

```

---

## 🔬 Methodology

### 1. Scalable Data Engineering

To handle the 220M+ event dataset within memory constraints, the pipeline utilizes **Polars** for parallelized processing:

* **Chunked Ingestion:** Raw JSONL files processed in 500k-row chunks.
* **Memory Optimization:** Downcasting numerical types (UInt32) and categorical encoding.
* **Storage:** Conversion to **Apache Parquet** format for 10x compression and fast columnar reads.

### 2. Stage I: Candidate Generation (High Recall)

Candidates are retrieved using Item-Based Collaborative Filtering via **Co-visitation Matrices**. Three distinct graph-based signals were engineered to capture different intent levels:

* **Click-to-Click:** Captures exploration behavior (24h window).
* **Click-to-Conversion:** Links exploration to high-intent actions (14-day window).
* **Buy-to-Buy:** Identifies complementary products frequently purchased together (7-day window).

### 3. Stage II: Ranking (High Precision)

The ranking stage treats the problem as a binary classification task using **Gradient Boosted Decision Trees (LightGBM)**.

* **Feature Engineering:** Over 60 features were engineered, including:
* *Session Context:* Session length, dwell time, recency scores.
* *Item Popularity:* Global click/cart/order rates.
* *Interaction Features:* Frequency of item interaction within the current session.


* **Multi-Task Strategy:** Three separate LightGBM models were trained specifically for Clicks, Carts, and Orders to handle the distinct class imbalances of each objective.

---

## 📊 Experiments & Results

### Validation Strategy

A time-based split was used, reserving the last 3 days of the dataset for validation to prevent look-ahead bias and simulate real-world forecasting.

### Performance Metrics

The system was evaluated using Recall@20.

| Metric | Score | Note |
| --- | --- | --- |
| **Public Score** | 0.42865 | Test set partition A |
| **Private Score** | **0.42946** | Test set partition B |

*The minimal gap between Public and Private scores demonstrates strong generalization and lack of overfitting.*

### Feature Importance Analysis

Interpretability analysis via LightGBM Gain revealed:

1. **Co-visitation Rank:** The strongest predictor across all tasks, validating the candidate generation strategy.
2. **Recency Score:** Highly influential for "Clicks," confirming that immediate user context drives exploration.
3. **Global Conversion Rate:** Critical for "Cart" and "Order" prediction, acting as a prior probability for purchase likelihood.

---

## 🛠 Usage

### Prerequisites

* Python 3.9+
* Polars
* LightGBM
* NumPy / Pandas

### Installation

```bash
git clone [https://github.com/hoanglechau/multi-objective-session-based-recommender-system-ml.git](https://github.com/hoanglechau/multi-objective-session-based-recommender-system-ml.git)

```

---

## 🔮 Future Work & Limitations

* **Cold Start:** The current co-visitation approach relies on history. Future iterations could incorporate **Graph Neural Networks (GNNs)** or Content-Based filtering to handle zero-shot item recommendations.
* **Sequential Modeling:** Replacing aggregated session features with **Transformer-based** architectures (e.g., SASRec) to better model the temporal sequence of events.

---

## 📚 References

* OTTO – Multi-Objective Recommender System (Kaggle Competition)
* Covington, P., et al. "Deep neural networks for youtube recommendations." RecSys 2016.
* Ke, G., et al. "LightGBM: A highly efficient gradient boosting decision tree." NeurIPS 2017.