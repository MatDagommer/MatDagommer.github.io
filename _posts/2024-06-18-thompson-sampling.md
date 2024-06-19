---
layout: post
title: Thompson Sampling
date: 2024-06-18 23:12:00
description: A method to accelerate ultra-large library screening
tags: Active Learning, Synthons, ML, Algorithm, Multi-armed bandit, Optimization
categories: ML
featured: true
---

{% include figure.liquid path="assets/img/thompson_sampling.jpeg" class="img-fluid rounded z-depth-1" %}


Screening of molecular libraries is usually performed to identify compounds with good properties. For moderately sized libraries (<$$10^9$$ compounds) and scoring functions that are fast at inference, it is possible to screen the entire library in a matter of hours. However, things get more complicated with ultra-large libraries (>$$10^9$$ compounds) and for scoring functions that take some time, like docking calculations.


In this [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790), the authors propose Thompson sampling (TS) as a technique to explore synthesis-on-demand databases, which are libraries containing all the reactants. These libraries are efficient because they don’t require all the products to be stored but offer the possibility to retrieve them by combining the reactants. For reactions with 2 reactants, with a library of $$10^6$$ reactants where all reactants can be paired, you save yourself the trouble of storing $$5 \times 10^11$$ molecules in memory. Practical.


**Note:** The combination of reactants is not trivial and requires some additional tools that 1. All reagents (“synthons”) are synthesizable or commonly available and 2. A method to predict the reaction product. This is what tools like the SynthOn library do (it is non-trivial, and would deserve a dedicated post). 


Similar to the UCB algorithm I discussed [previously](https://matdagommer.github.io/blog/2024/bayes-UCB-algorithm/), TS is a Bayesian approach that can be used to solve the multi-armed bandit problem: in the face of multiple choices (arms), TS samples rewards from the reward distribution of each arm. The arm with the maximum sampled reward gets selected, and its reward distribution gets updated after scoring (Bayesian belief update, posterior distribution).


The cool thing the paper does is tweak TS slightly to make it work for library screening. Each arm represents a reactant, and an action consists of selecting $$n$$ arms for a reaction with $$n$$ reactants. The reactants are selected by taking the $$n$$ highest sampled rewards among the $$N$$ arms. After that, the reactants are combined, and the resulting product is scored. The trick is that the $$n$$ arms’ distributions are updated with that same score.


This way, you can develop individual reward distributions for each reactant that indicate whether a reactant is likely to give satisfactory products when combined with other molecules. Through multiple iterations, the algorithm tries and tests combinations that are more likely to yield satisfactory results, which can help constitute a library of good candidates.


**Remark:** In their implementation, the authors use Gaussian priors with fixed variance for the reward distributions. This is the same distribution used in the Bayes-UCB algorithm from Wang et al.


The authors tested it on three datasets for three different problems: a Tanimoto similarity search using the Niementowski quinazoline library, a 3D similarity search using the Enamine REAL collection, and a docking prediction using the Enamine library **m_22bba**. In each case, the authors were able to screen less than 1% of possible reagent combinations and still retrieve over 50% of the best compounds (the assessment was possible by conducting an exhaustive search on the side).


In this [preprint](https://www.biorxiv.org/content/10.1101/2024.05.16.594622v1.full.pdf), Zhao et al. propose stochastic wheel roulette selection to improve the performance of TS. Their idea consists of converting reagents' samples into probabilities using the Boltzmann formula and then sampling the reagent space using this new probability. It proved quite successful on the Niementowski quinazoline library for Tanimoto search: by exploring 0.01% of the library, the authors were able to retrieve 100% of the hits (versus 90% for TS). The method has yet to be tested on the other two, more challenging datasets but this is a promising start.

