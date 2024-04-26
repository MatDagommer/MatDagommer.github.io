---
layout: post
title: Decision trees in 5 minutes
date: 2024-04-25 15:09:00
description: Learn how Decision Trees are trained.
tags: Decision Tree, ML, Random Forest, XGBoost
categories: ML
featured: true
---

After 2 years of doing projects in the machine learning space, I realized I didn't know how decision tree models actually worked:
although decision trees have been around forever (on the scale of ML history), I never felt the urge of learning the inner working for several reasons:
- My early ML classes focused on differentiable models (Andrew Ng).
- Implementation of decision trees with the Python scikit-learn, is very easy and high-level.
- Although I trained decision trees as benchmarks, I never thought of them as a go-to models.

Well, I (finally) decided to take a good look! In this short post, I try to describe the training process of a decision tree in an accessible language, as a reference for myself and for folks looking for a concise, yet technical explanation.

I. The tree structure

{% include figure.liquid path="assets/img/9.jpg" class="img-fluid rounded z-depth-1" %}

I'm sure you've encountered tree structures before, and they're fairly easy to make sense of when presented in that format: take your input, and depending on its features, follow the path. Boom. Your output is either a class (classification), or value (regression). Straightforward. 

{% include figure.liquid path="assets/img/decision_tree.jpg" class="img-fluid rounded z-depth-1" %}

But this tree is just one tree among a myriad of possibilities, and I could come up with an infinity of combinations: why not sort wines based on their acidity instead of their alcohol content? Why not change the threshold to 0.4 instead of 0.6? What decided that structure? Let's break it down:

At every node (nodes represent stages at which data gets sorted) of the tree, we want to separate the training data into different groups in a fashion that allows us to retrieve more homogeneous subsets at every step. In the case of regression, this means grouping training points with similar target values. In the case of classification, this means grouping points with similar classes. In order to determine what feature will be used as a criteria for the separation, and what threshold will be used, the algorithm loops through all possible (feature, threshold) combinations and chooses the best one based on a criterion metric (information gain, Gini impurity, variance reduction). This criterion metrics assess if the two subgroups are more homogeneous than the initial group.

Let's take the information gain as an example: 

Without diving into too much details, Gini and information gain can be used for assessing this in the case of classification problems, while variance reduction is designed for regression problems. The conclusion is: we have objective, quantitative ways of assessing what (feature, threshold) combination returns the most information at each node. I leave you with this intuition, but I encourage you to check the math behind these metrics!

What about cases where there are more than 2 branches? Usually, one can set a maximum number of leaves per node. The algorithm proceeds the same way, but the loop depth increases (feature, threshold_1, threshold_2, ..., threshold_n) combinations and find the one that returns the maximum information gain. 

What about categorical features? In the case of categorical features, we loop through the categories instead of threshold values: (feature, cat U not(cat)). 

What about continuous features: there could be an infinite number of thresholds to loop through!

What about the depth?

Models based on decision trees:

Random forests:
XGBoost:

Features
Structure
Criterion
Gini
Variance Reduction
Leaves
Nodes
Constraints


````markdown
```c++
code code code
```
````

```c++
int main(int argc, char const \*argv[])
{
    string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;
}
```

For displaying code in a list item, you have to be aware of the indentation, as stated in this [Stackoverflow answer](https://stackoverflow.com/questions/34987908/embed-a-code-block-in-a-list-item-with-proper-indentation-in-kramdown/38090598#38090598). You must indent your code by **(3 \* bullet_indent_level)** spaces. This is because kramdown (the markdown engine used by Jekyll) indentation for the code block in lists is determined by the column number of the first non-space character after the list item marker. For example:

````markdown
1. We can put fenced code blocks inside nested bullets, too.

   1. Like this:

      ```c
      printf("Hello, World!");
      ```

   2. The key is to indent your fenced block in the same line as the first character of the line.
````

Which displays:

1. We can put fenced code blocks inside nested bullets, too.

   1. Like this:

      ```c
      printf("Hello, World!");
      ```

   2. The key is to indent your fenced block in the same line as the first character of the line.

By default, it does not display line numbers. If you want to display line numbers for every code block, you can set `kramdown.syntax_highlighter_opts.block.line_numbers` to true in your `_config.yml` file.

If you want to display line numbers for a specific code block, all you have to do is wrap your code in a liquid tag:

{% raw %}
{% highlight c++ linenos %} <br/> code code code <br/> {% endhighlight %}
{% endraw %}

The keyword `linenos` triggers display of line numbers.
Produces something like this:

{% highlight c++ linenos %}

int main(int argc, char const \*argv[])
{
string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;

}

{% endhighlight %}
