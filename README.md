# Kaggle Projects

This repository contains personal projects associated with Kaggle competitions and datasets.

## [Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations) (Dec 2022 - Mar 2023)

The goal of this competition was to streamline the process of matching educational content (documents, videos, webpages, etc.) to specific topics in a curriculum (a collection of hierarchical subject taxonomies in various languages). The competition was hosted by the non-profit organization Learning Equality, together with The Learning Agency Lab and UNHCR. Efficient and scalable solutions would support efforts to help people across the world access quality education, providing curricular experts with tailored recommendations for open educational resources relevant to local school programs, therefore reducing their time spent curating content.

See my public submission [here](https://www.kaggle.com/federicodevitohalevy/lecr-modeling). (Note that hyperparameters may not match those described below.)

### Introduction

**Framing:** The challenge of curriculum alignment, although difficult, is similar to that of recommender, document retrieval and question answering systems, with some key characteristics: 1. Topic-topic relations (analogous to, e.g., user-user interactions) form a graph containing hierarchical and disconnected components; 2. There are no explicit content-content (i.e. item-item) interactions; 3. Target topic-content correlations (*a*) are sparse but quite strictly aligned (as opposed to many recommenders), (*b*) support a range of possible numbers of matches for any given item which includes none (in contrast to, e.g., document retrieval), and (*c*) do not necessarily involve directly inter-quotable passages (as, for example, found in many question-answer contexts).

**Strategy:** My proposed solution approaches the task as a metric learning and vector search problem, and generates content recommendations in two stages. Each topic and content is assigned a vector representation---an embedding of its text features---in a single shared latent space, such that neighboring items (of distinct type) are likely to be good matches. For the similarity search to be scalable, a pool of candidate pairs is first retrieved by comparing bi-encoded embeddings, learned by two transformer networks attending separately to topics and contents (forming a `retriever` model). A given topic and up to `k` of its nearest content neighbors are then passed to a third network (a `reranker` model), which attends to cross-encoded features before classifying each pairing either as a match or a suggestion to reject.

**Related work:** The implementation combines ideas from the information retrieval literature, with the following key references: [Karpukhin et al.](https://arxiv.org/abs/2004.04906) (retriever setup: biencoding, similarity metric, loss function, in-batch negatives), [Hamilton et al.](https://arxiv.org/abs/1706.02216) (graph learning: aggregation algorithm), [Glass et al.](https://arxiv.org/abs/2207.06300) (multi-stage setup: reranker architecture, reranker loss function, staged training), and the `sentence-transformers` [codebase](https://github.com/UKPLab/sentence-transformers) and [documentation](https://www.sbert.net/).

### Exploratory analysis

**Data:** The inputs involved are two tabular datasets---one for topics, another for contents, with each describing at least ~100k items---consisting mostly of text fields (with few numerical and categorical fields). The target is a set of well-aligned topic-content pairs, which does not necessarily feature all input topics and contents. A training set of ~80k topics, ~150k contents and ~280k positive labels for correlated topic-content pairings was provided, allowing for supervised learning.

**Evaluation:** The evaluation metric was mean F2 score averaged across topics in a hidden test set. Recall is given more weight than precision given the importance of not missing relevant content when providing experts with recommendations to refine.

See my [EDA notebook](https://github.com/FdVH/kaggle/tree/master/learning-equality-curriculum-recommendations/lecr-exploration.ipynb) for more details.

### Modeling

**Pre-processing:** After minimal text cleaning, re-indexing and conversion of non-text fields, tokenized representations of topics and contents are created by concatenating and encoding text fields. These are truncated or padded to a fixed length, and stored in memory-mapped files for efficient access. A neighbor sampler of the topic graph is also constructed, which navigates an undirected edge list corresponding to the parent/child references of the input taxonomy.

**Embedding backbone:** Token sequences are first embedded into dense vectors using a pre-trained LLM, tuned specifically for sentence-level semantic similarity tasks. The final model used, in particular, is [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)---a multi-lingual `SBERT` model containing ~117M parameters and 384 hidden dimensions. This architecture is used as a backbone for all three encoder models.

**Embedding adjustment:** Topic and content embeddings are obtained with distinct backbone instances. These are then rotated and adjusted by smaller model blocks. The topic encoder exploits knowledge of the topic graph to inform the embedding of each topic, by sampling and aggregating those of its neighbors via several graph convolutional layers. The simpler content encoder, on the other hand, implements two full-connected layers for this. Both use GeLU activation and dropout regularization in hidden layers.

**Candidate retrieval:** After biencoding, a top-*k* content search for cosine similarity with each topic is executed in the corpus of contents that match a given topic's language. This is implemented either with an exact tensor inner product, or using a quantized vector index when the size of the content corpus makes the former infeasible. Of the resulting candidates for any given topic, only those with a similarity score within a fixed margin of the respective top score are kept. Stringent filtering at this stage is crucial to avoid processing too many candidates with the reranker's costly cross-encoder at the next stage.

**Candidate reranking:** The tokens of each retrieved candidate pairing are crossed and processed together by a third backbone instance with a classification head (that is not pre-trained). Outputs are interpreted as log-scale confidence scores and filtered by a fixed threshold to produce the final predictions. 

**Software:** The code is `PyTorch`-based and depends (among others) on `transformers` and `PyG` (PyTorch Geometric) for embedding, `Faiss` for search and `Pandas` for data manipulations. Custom routines employ various techniques to optimize for the memory, compute and runtime available for the project, including: mixed-precision operations, memory-mapping pre-encoded tokens, vectorized batch and graph operations (e.g. for cross-encoding, neighbor sampling, computing similarity, formatting, etc.), dynamic (fixed margin) thresholding, staged network training, gradient checkpointing and acculation, and quantized vector search (for large content corpora).

See my [modeling notebook](https://github.com/FdVH/kaggle/tree/master/learning-equality-curriculum-recommendations/lecr-modeling.ipynb) for more details.

### Training

**Datasets:** (*a*) *Retriever*. Positive samples are randomly chosen among ground truth topic-content pairs such that at least one positive sample exists per topic in any given batch. Negatives are generated in-batch and consist of all non-correlated topic-content pairings that can be defined in the set of items considered for a given batch of positive samples. (*b*) *Reranker*. All ground truth pairs are used as positive samples. These are augmented with negatives sampled randomly among all other possible pairings, such that the resulting positive-to-negative ratio is 2:1. 

**Objective:** (*a*) *Retriever*. The embedding model learns (in a contrastive manner) to minimizing the negative log-likelihood of positive biencoded pairs, defined by their softmaxed inner product, normalized over all in-batch positives and negatives for the respective topic. (*b*) *Reranker*. Loss is defined as the binary cross-entropy between predicted confidence scores and true labels.

**Schedule:** (*a*) *Retriever*. First, the two backbones are fine-tuned with the competition data for 10 epochs. Second, the adjustment blocks---initialized to approximate the identity mapping---are trained while freezing the backbone for 10 epochs. (*b*) *Reranker*. Learned weights from the retriever are not currently transferred over. Instead, the pre-trained backbone weights and randomly initialized classification head are trained together for 10 epochs. (*c*) *Global*. All training rounds follow a one-cycle learning rate schedule over all epochs to improve convergence times: starting end ending at low rates and gradually rising to/falling from an intermediary peak rate at each update step. Updates occur after gradients have accumulated over 8 batches of 512 topics, effectively simulating a large batch size of 4096 topics for improved contrastive learning. No end-to-end training is currently implemented.

**Validation:** During development, topics were split for k-fold cross validation while preserving the hierarchical structure of the taxonomies. Models were trained with early stopping based on the loss for ~1000--4000 validation topics, and evaluated by tracking recall, precision and F2 score, and inspecting activation maps (particularly for the adjustment blocks). The final model was trained on all available competition data.

**Hardware:** Training was performed with single-GPU virtual machines on Kaggle (1 x `P100` 16GB, mostly for development) and Google Cloud Platform (1 x `A100` 40GB).

### Future work

- Stacking biencoded and cross-encoded representations of topic-content pairs before the reranker's classification head (a new layer with 3 input channels) would introduce gradient flow to the retriever (with end-to-end training) and might improve performance.

- Text augmentations (e.g. multiple crops, translated samples, etc.) for larger and more robust training.