# Gaussian Processes

## Introduction
I just delved into this cool thing called Gaussian processes and I thought I would make a quick summary of what I was able to grasp from the concept. I came across Gaussian processes in the past in the context of hyperparameter optimization but never dug further. But a more global reflection on uncertainty quantification in machine learning led me to take a closer look. Turns out that Gaussian processes inherently allow us to compute some kind of uncertainty!

## Bayesian Perspective on Weight Derivation
What most machine learning models do is come up with a set of weights or a model we can use to do inference. Usually, the way these weights are derived is by looking at the Maximum A Posteriori (MAP), which means maximizing the posterior distribution. If you look at the Bayesian interpretation of ordinary least squares or ridge regression, it turns out that coming up with those weights consists in optimizing or retrieving weights that maximize the posterior distribution of the weights.

\[
\hat{\beta} = \arg\max_\beta P(\beta | X, Y) = \arg\max_\beta P(Y | X, \beta) P(\beta)
\]

## Posterior Predictive Distribution
But the problem with such an approach is that you only consider one set of weights for your predictions. And in the process, you ruled out completely other possible models, although they may be as valid! Gaussian processes allow us to take into account all model configurations (yes sir!) by directly modeling the posterior predictive distribution. If we look at it from the Bayesian perspective, we will see that the posterior predictive distribution integrates over all possible sets of model weights:

\[
P(Y^* | X^*, X, Y) = \int P(Y^* | X^*, \beta) P(\beta | X, Y) d\beta
\]

## Assumption of Gaussian Distributions
The idea behind Gaussian processes is to assume that the distributions from our problem are Gaussian. This will allow us to make the assumption that the posterior predictive distribution is also Gaussian! This video explains nicely why this assumption is reasonable. Here is the short version: if you look at the formula below and if everything on the right is assumed to be Gaussian, then the posterior distribution becomes Gaussian immediately (because Gaussian distributions are conjugate prior of… Gaussian distributions).

\[
P(Y | X) \sim \mathcal{N}(\mu, \Sigma)
\]

## Covariance Matrix and Kernel Trick
Now that we have shown that it is reasonable, we'll just assume it from now on. Now here's the trick: not only do we assume that the training set follows a Gaussian distribution, we also assume that the combination of the training points and the test points also follow a Gaussian distribution.

The question becomes, what should we take as our Gaussian distribution parameters? That is, our mean values and covariance matrix. It turns out we don't really care about the mean values. The simple, practical explanation is that we can center the data around zero as a preprocessing step. Regarding the covariance matrix, we can use a cool trick: a kernel! Essentially, a kernel enables us to define a surrogate for the covariance matrix which is more a similarity matrix than anything. 

\[
\Sigma = K(X, X)
\]

The RBF kernel is commonly used and is a way to capture the similarity between two data points:

\[
K(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)
\]

## Customizing Covariance Matrix
You may ask, “Why the heck are we customizing the covariance matrix?” Here’s why: there is no way we can come up with a meaningful covariance matrix given our current dataset. What would it even mean to compute a variance between \(y_1\) and \(y_n\)? At the end of the day, they correspond to the same quantity, just conditioned differently! However, we can use a kernel function to compute the similarity between input values. And in return, these can lead to a surrogate to the covariance matrix that makes sense: if two input values \(x\) and \(x'\) are very similar, then we expect our model to provide output values \(y\) and \(y'\) that are similar too! This should translate as a high correlation in the joint distribution between the two variables. High similarity, high correlation!

By choosing an appropriate kernel function that correctly captures the similarity between input values, we should retrieve a nice joint distribution. Now, we need to derive the conditional distribution of the test point we want to infer. Here’s the formula, I’m sparing myself the math:

\[
P(Y^* | X^*, X, Y) \sim \mathcal{N}(K(X^*, X) K(X, X)^{-1} Y, K(X^*, X^*) - K(X^*, X) K(X, X)^{-1} K(X, X^*))
\]

## Conclusion
This is only the beginning of my exploration with Gaussian processes, and I'm eager to learn more about the applications. My understanding is that Gaussian processes are well suited for small datasets but scale badly with big datasets, especially because matrix inversion has a computation time complexity of \(O(n^3)\). However, I am really hyped about the uncertainty measure we can get out of it.


