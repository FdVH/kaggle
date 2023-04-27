# Kaggle Projects

This repository contains personal projects associated with Kaggle competitions and datasets.

## [Learning Equality - Curriculum Recommendations](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations) (Dec 2022 - Mar 2023)

The goal of this competition was to streamline the process of matching educational content (documents, videos, webpages, etc.) to specific topics in a curriculum (a collection of hierarchical subject taxonomies in various languages). The competition was hosted by the non-profit organization Learning Equality, together with The Learning Agency Lab and UNHCR. Efficient and scalable solutions would support efforts to help people across the world access quality education, providing curricular experts with tailored recommendations for open educational resources relevant to local programs, and therefore reducing time spent curating content.

See my public submission [here](https://www.kaggle.com/federicodevitohalevy/lecr-modeling).

### Exploration

**Data:** The inputs involved were two tabular datasets---one for topics, another for contents---consisting mostly of text fields (with few numerical and categorical fields).

**Evaluation:** The evaluation metric was mean F2 score averaged across topics in a hidden test set.

See my [EDA notebook](https://github.com/FdVH/kaggle/tree/master/learning-equality-curriculum-recommendations/lecr-exploration.ipynb) for details.

### Modelling

**Pre-processing:** After some text cleaning and conversion of non-text fields, tokenized representations of topics and contents are created by concatenating and encoding text fields.

**Embedding:** Token sequences are first embedded into dense vectors using a pre-trained multi-lingual LLM (specifically, 'paraphrase-xxx'), fine-tuned .

**Search:** Content recommendations are generated using a two-stage approach: retrieve and rerank. 

**Training:** Topics are split for k-fold validation while preserving the hierarchical structure of the taxonomies. The model is trained for 10 epochs, with checkpointing based on the validation loss.

See my [modeling notebook](https://github.com/FdVH/kaggle/tree/master/learning-equality-curriculum-recommendations/lecr-modeling.ipynb) for details.


### Implementation details

Various techniques were employed to optimize for the available memory, compute and runtime, including: mixed-precision operations, memory-mapping pre-encoded tokens, vectorized batch and graph operations (e.g. for cross-encoding, neighbor sampling, embedding, computing similarity scores, formatting, etc.), quantized vector search (for cases with large content corpora), gradient checkpointing