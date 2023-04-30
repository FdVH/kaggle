# Kaggle Projects

This repository contains personal projects associated with Kaggle competitions and datasets.

## [Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations) (Dec 2022 - Mar 2023)

The goal of this competition was to streamline the process of matching educational content (documents, videos, webpages, etc.) to specific topics in a curriculum (a collection of hierarchical subject taxonomies in various languages). The competition was hosted by the non-profit organization Learning Equality, together with The Learning Agency Lab and UNHCR. Efficient and scalable solutions would support efforts to help people across the world access quality education, providing curricular experts with tailored recommendations for open educational resources relevant to local school programs, therefore reducing their time spent curating content.

See my public submission [here](https://www.kaggle.com/federicodevitohalevy/lecr-modeling).

### Introduction

**Framing:** The challenge of curriculum alignment, although difficult, is similar to that of recommender, document retrieval and question answering systems, with some key characteristics: 1. Topic-topic relations (analogous to, e.g., user-user interactions) form a graph containing hierarchical and disconnected components; 2. There are no explicit content-content (i.e. item-item) interactions; 3. Target topic-content correlations (*a*) are sparse but quite strictly aligned (as opposed to many recommenders), (*b*) support a range of possible numbers of matches for any given item which includes none (in contrast to, e.g., document retrieval), and (*c*) do not necessarily involve directly inter-quotable passages (as, for example, found in many question-answer contexts).

**Strategy:** My proposed solution approaches the task as a metric learning and vector search problem, and generates content recommendations in two stages. Each topic and content is assigned a vector representation---an embedding of its text features---in a single shared latent space, such that neighboring items (of distinct type) are likely to be good matches. For the similarity search to be scalable, a pool of candidate pairs is first retrieved by comparing bi-encoded embeddings, learned by two transformer networks attending separately to topics and contents (forming a `retriever` model). A given topic and up to `k` of its nearest content neighbors are then passed to a third network (a `reranker` model), which attends to cross-encoded features before classifying each pairing either as a match or a suggestion to reject.

**Related work:** The implementation combines ideas from the information retrieval literature, with the following key references: [Karpukhin et al.](https://arxiv.org/abs/2004.04906) (retriever setup: biencodings, similarity metric, loss function, in-batch negatives), [Hamilton et al.](https://arxiv.org/abs/1706.02216) (graph learning: aggregation algorithm), [Glass et al.](https://arxiv.org/abs/2207.06300) (multi-stage setup: reranker architecture, reranker loss function, staged training), ... .

### Exploratory analysis

**Data:** The inputs involved are two tabular datasets---one for topics, another for contents, with each describing at least ~100k items---consisting mostly of text fields (with few numerical and categorical fields). The target is a set of well-aligned topic-content pairs, which does not necessarily feature all input topics and contents. A training set of ~80k topics, ~150k contents and ~280k positive labels for correlated topic-content pairings was provided, allowing for supervised learning.

**Evaluation:** The evaluation metric was mean F2 score averaged across topics in a hidden test set. Recall is given more weight than precision given the importance of not missing relevant content when providing experts with recommendations to refine.

See my [EDA notebook](https://github.com/FdVH/kaggle/tree/master/learning-equality-curriculum-recommendations/lecr-exploration.ipynb) for more details.

### Modeling

**Pre-processing:** After minimal text cleaning, re-indexing and conversion of non-text fields, tokenized representations of topics and contents are created by concatenating and encoding text fields. These are truncated or padded to a fixed length, and stored in memory-mapped files for efficient access. A neighbor sampler of the topic graph is also constructed, which navigates an undirected edge list corresponding to the parent/child references of the input taxonomy.

**Embedding backbone:** Token sequences are first embedded into dense vectors using a large language model, pre-trained specifically for sentence-level semantic similarity tasks. The final model used, in particular, is `paraphrase-multilingual-MiniLM-L12-v2`, a multi-lingual `SBERT` model containing ~110M parameters and 384 hidden dimensions. This architecture is used as a backbone for all three encoder models.

**Embedding adjustment:** Embeddings from the backbone are then rotated and adjusted by a smaller model block. The topic encoder exploits knowledge of the topic graph to inform the embedding of each topic, by aggregating those of its neighbors via several graph convolutional layers. The simpler content encoder, on the other hand, implements a single full-connected layer for this.

**Search:** Content recommendations are generated using a two-stage approach. *Retrieval* (stage 1):  retrieve and rerank.

**Software:** The code is `PyTorch`-based and depends (among others) on `HuggingFace transformers` and `PyTorch Geometric` for embedding, `Faiss` for search and `Pandas` for data manipulations. Custom routines employ various techniques to optimize for the memory, compute and runtime available for the project, including: mixed-precision operations, memory-mapping pre-encoded tokens, vectorized batch and graph operations (e.g. for cross-encoding, neighbor sampling, computing similarity, formatting, etc.), staged network training, gradient checkpointing and acculation, quantized vector search (for large content corpora), ... .

See my [modeling notebook](https://github.com/FdVH/kaggle/tree/master/learning-equality-curriculum-recommendations/lecr-modeling.ipynb) for more details.

### Training

**Dataset:** ... Retriever: positive samples are randomly chosen among ground truth topic-content pairs such that at least one positive sample exists per topic in any given batch; negatives are generated in-batch and consist of all non-correlated topic-content pairings that can be defined in the set of items considered for a given batch of positive samples. Reranker: all ground truth pairs are used as positive samples and are augmented with negatives sampled randomly from among all other possible pairings. ...

**Objective:** The retriever's embedding model is trained in a contrastive manner to minimize the negative log-likelihood of the inner product between positive biencoded pairs. Reranker: binary cross-entropy loss.

**Schedule:** ... Retriever: first, the backbones are fine-tuned with the competition data for 10 epochs; second, the adjustment blocks are initialized to approximate the identity mapping and are trained while freezing the backbone for 10 epochs. Reranker: the ... . All training rounds follow a one-cycle learning rate schedule over all epochs, starting at low rates and gradually rising/falling at each update step. These occur after gradients have accumulated over 8 batches of 512 topics.

**Validation:** Topics are split for k-fold cross validation while preserving the hierarchical structure of the taxonomies. The model is trained for 10 epochs, with checkpointing based on the validation loss.

**Hardware:** Training was performed on single-GPU virtual machines from Kaggle (1 x `P100` 16GB during development) and Google Cloud Platform (1 x `A100` 40GB for final model).

### Future work

- Stacking the biencoded and cross-encoded representations of topic-content pairs and training a new classification head for the reranker (with 3 input channels) might improve performance.