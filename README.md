# N-Gram Language Modeling

## Overview

This project explores statistical language modeling through the implementation of N-gram models in Python. The objective is to estimate the probability of word sequences, evaluate model performance using perplexity, and generate new text based on learned word distributions.

N-gram models provide a probabilistic approach to natural language processing (NLP) by approximating the probability of a word using a fixed number of preceding words.

## Mathematical Formulation

### 1. N-Gram Language Model

For a sequence of words $w_1, w_2, \ldots, w_T$, the chain rule of probability gives:

```math
P(w_1, w_2, \ldots, w_T)
=
\prod_{i=1}^{T} P(w_i \mid w_1, \ldots, w_{i-1})
```

An N-gram model makes the Markov assumption that the probability of the next word depends only on the preceding $n-1$ words:

```math
P(w_i \mid w_1, \ldots, w_{i-1})
\approx
P(w_i \mid w_{i-n+1}, \ldots, w_{i-1})
```

### 2. Maximum Likelihood Estimation

Conditional probabilities are estimated from observed N-gram frequencies:

```math
P(w_i \mid w_{i-n+1}, \ldots, w_{i-1})
=
\frac{C(w_{i-n+1}, \ldots, w_i)}
{C(w_{i-n+1}, \ldots, w_{i-1})}
```

where $C(\cdot)$ represents the frequency of a word sequence in the training corpus.

### 3. Backoff Probability Estimation

When an N-gram context has not been observed in the training data, the model can back off to a shorter context to estimate the next-word probability.

This allows the model to use lower-order N-gram statistics when higher-order statistics are unavailable.

### 4. Perplexity

Perplexity measures how well a language model predicts a sequence of words. For $T$ predicted tokens, it is defined as:

```math
\operatorname{PP}(W)
=
\exp\left(
-\frac{1}{T}
\sum_{i=1}^{T}
\log P(w_i \mid w_{<i})
\right)
```

Lower perplexity indicates better predictive performance on the evaluation data, when models are compared under the same evaluation conditions.
