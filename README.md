# NLP project, Open University of Israel, Spring 2019

## Prerequisites
1. Python 3.7
1. All packages mentioned in [requirements.txt](requirements.txt)

## Aspect Extraction
See the notebooks in [Aspect_Extraction](Aspect_Extraction)

## Sentiment Analyser
* Data is in the [data](data) directory
* The predictor itself can be run by executing [predictor.py](predictor.py)

---
Following is the README of the forked repository
---

# Neural Sentiment Analyzer for Modern Hebrew


This code and dataset provide an established benchmark for neural sentiment analysis for Modern Hebrew.
For more information, and for attribution, please refer to Amram, A., Ben-David, A., and Tsarfaty, R. (2018). [Representations and Architectures in Neural Sentiment Analysis for Morphologically Rich Languages: A Case Study from Modern Hebrew](http://aclweb.org/anthology/C18-1190). In: Proceedings of The 27th International Conference on Computational Linguistics (COLING 2018) Santa Fe, NM, (pp. 2242-2252).

## Abstract
This paper empirically studies the effects of representation choices on neural sentiment analysis for Modern Hebrew, a morphologically rich language (MRL) for which no sentiment analyzer currently exists. We study two dimensions of representational choices: (i) the granularity of the input signal (token-based vs. morpheme-based), and (ii) the level of encoding of vocabulary items (string-based vs. character-based). We hypothesise that for MRLs, languages where mul- tiple meaning-bearing elements may be carried by a single space-delimited token, these choices will have measurable effects on task perfromance, and that these effects may vary for different architectural designs: fully-connected, convolutional or recurrent. Specifically, we hypothesize that morpheme-based representations will have advantages in terms of their generalization capac- ity and task accuracy, due to their better OOV coverage. To empirically study these effects, we develop a new sentiment analysis benchmark for Hebrew, based on 12K social media comments, and provide two instances thereof: token-based and morpheme-based. Our experiments show that the effect of representational choices vary with architectural types. While fully-connected and convolutional networks slightly prefer token-based settings, RNNs benefit from a morpheme- based representation, in accord with the hypothesis that explicit morphological information may help generalize. Our endeavor also delivers the first state-of-the-art broad-coverage sentiment analyzer for Hebrew, with over 89% accuracy, alongside an established benchmark to further study the effects of linguistic representation choices on neural networks’ task performance.
