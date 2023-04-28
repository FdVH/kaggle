# Kaggle Projects

This repository contains personal projects associated with Kaggle competitions and datasets.

## [Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations) (Dec 2022 - Mar 2023)

The goal of this competition was to streamline the process of matching educational content (documents, videos, webpages, etc.) to specific topics in a curriculum (a collection of hierarchical subject taxonomies in various languages). The competition was hosted by the non-profit organization Learning Equality, together with The Learning Agency Lab and UNHCR. Efficient and scalable solutions would support efforts to help people across the world access quality education, providing curricular experts with tailored recommendations for open educational resources relevant to local programs, and therefore reducing time spent curating content.

See my public submission [here](https://www.kaggle.com/federicodevitohalevy/lecr-modeling).

### Exploration

**Data:** The inputs involved were two tabular datasets---one for topics, another for contents---consisting mostly of text fields (with few numerical and categorical fields). The targets were ... . Training set ... allowed for supervised learning.

**Evaluation:** The evaluation metric was mean F2 score averaged across topics in a hidden test set.

**Framing:** The challenge of generating content recommendations for a given topic, although difficult, is similar to that of recommender, document retrieval and question answering systems, with a few key characteristics: 1. Topic-topic relations (analogous to, e.g., user-user interactions) form a graph with hierarchical and disconnected components. 2. There are no explicit content-content (i.e. item-item) interactions. 3. Target topic-content correlations are sparse but quite strictly aligned (as opposed to many recommenders), support a range of possible numbers of matches which includes none (in contrast to, e.g., document retrieval), and do not necessarily involve directly inter-quotable passages (as, for example, found in many question-answer contexts).

**Strategy:** My proposed solution approaches the task as a combined metric learning and vector search problem. ...

**Related work:** My implementation combines ideas from the information retrieval literature, with the following key references: [Karpukhin et al.](https://arxiv.org/abs/2004.04906) (retriever setup: biencodings, similarity metric, loss function, in-batch negative samples), [Hamilton et al.](https://arxiv.org/abs/1706.02216) (graph learning: aggregation algorithm), [Glass et al.](https://arxiv.org/abs/2207.06300) (multi-stage setup: reranker architecture, reranker loss function, staged training), ... .

See my [EDA notebook](https://github.com/FdVH/kaggle/tree/master/learning-equality-curriculum-recommendations/lecr-exploration.ipynb) for more details.

### Modeling

**Pre-processing:** After some text cleaning and conversion of non-text fields, tokenized representations of topics and contents are created by concatenating and encoding text fields.

**Embedding:** Token sequences are first embedded into dense vectors using a pre-trained multi-lingual LLM (specifically, 'paraphrase-xxx'), fine-tuned .

**Cross-encoding:** For the algorithm to be scalable, it is infeasible to cross-encode all possible topic-content pairs. 

**Search:** Content recommendations are generated using a two-stage approach. *Retrieval* (stage 1):  retrieve and rerank.

**Software:** The code is `PyTorch`-based. It uses `HuggingFace transformers` and `PyTorch Geometric for embedding, `Faiss` for search, `Pandas` for data manipulations, and implements custom routines for staged training and vectorized batch processing (among others).

### Training

**Dataset:** ... Retriever: positive samples are sampled from ground truth topic-content pairs such that at least one positive sample exists per topic in any batc; negatives are generated in-batch. Reranker: all ground truth pairs are used as positive samples and are augmented with negatives sampled randomly from among all possible pairings. ...

**Validation:** Topics are split for k-fold cross validation while preserving the hierarchical structure of the taxonomies. The model is trained for 10 epochs, with checkpointing based on the validation loss.

**Hardware:** Training was performed on single-GPU virtual machines from Kaggle (1 x `P100` 16GB during development) and Google Cloud Platform (1 x `A100` 40GB for final model).

See my [modeling and training notebook](https://github.com/FdVH/kaggle/tree/master/learning-equality-curriculum-recommendations/lecr-modeling.ipynb) for more details.


### Implementation details

Various techniques were employed to optimize for available memory, compute and runtime, including: mixed-precision operations, memory-mapping pre-encoded tokens, vectorized batch and graph operations (e.g. for cross-encoding, neighbor sampling, computing similarity, formatting, etc.), quantized vector search (for large content corpora), network gradient checkpointing