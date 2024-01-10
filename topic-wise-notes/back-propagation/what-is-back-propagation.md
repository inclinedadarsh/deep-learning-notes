# What is back propagation?

## Overview

> In machine learning, back-propagation is a gradient estimation method used to train neural network models. The gradient estimate is used by the [optimization algorithm](https://en.wikipedia.org/wiki/Optimization\_algorithm) to compute the network [parameter](https://en.wikipedia.org/wiki/Parameter\_\(machine\_learning\)) updates.
>
> _- Wikipedia_

Basically, used to train a neural network.

<img src="../../.gitbook/assets/file.excalidraw (1).svg" alt="Simple neural network" class="gitbook-drawing">

_We'll be referencing the above diagram throughout the notes_

So in above neural network, each arrow represents a weight and each perceptron has a bias. So basically, <mark style="color:yellow;">back propagation algorithm is used to find best values of all the weights and biases so that the neural network performs as good as it can</mark>.&#x20;

For this setting, we'll assume all the activation functions to be _linear_.

## Steps in back propagation

<table data-full-width="false"><thead><tr><th>IQ</th><th>CGPA</th><th>Package (LPA)</th></tr></thead><tbody><tr><td>80</td><td>8</td><td>5</td></tr><tr><td>60</td><td>5</td><td>3</td></tr><tr><td>110</td><td>7</td><td>8</td></tr></tbody></table>

Consider the above table as an example

1. Initialize all the weights and biases\
   Either make them random\
   Or initialize all the weights to be 1 and biases to be 0 (we'll do this)
2. Select a random student (row)\
   We're selecting the first row, whose package is 5
3. Predict the Package (target column) using dot product\
   As the initial weights are very random, the result will be incorrect\
   Suppose the answer we got is 17
4.  Choose a loss function\
    We're choosing MSE[^1] -> $$L = (y-\hat{y})^2$$\
    In our case, the answer will be $$L = (5 - 17)^2 = 144$$


5. Update the weights and biases using gradient descent using the following formula:

$$
w_{new} = w_{old}  -  \eta\frac{{\partial L}}{{\partial w_{old}}}
$$

$$
b_{new} = b_{old}  -  \eta\frac{{\partial L}}{{\partial b_{old}}}
$$

&#x20;        Where $$\eta$$ = Learning rate

## Hierarchy

So now the final output $$\hat{y}$$ or $$O_{21}$$ depends on numerous variables.\
$$O_{21} = w^2_{11} \cdot O_{11} + w^2_{21} \cdot O_{12} + b_{21}$$

Here, $$O_{21}$$ depends upon 5 variables, i.e. $$w^2_{11}, w^2_{21}, O_{11}, O_{12}, b_{21}$$. Now the weights $$w^2_{11}$$ & $$w^2_{21}$$ and bias $$b_{21}$$ are simple variables, however, the results $$O_{11}$$ & $$O_{12}$$ depend on their weights and biases. In this way, a hierarchy is formed.

During the whole algorithm, we select a random student (row) and then we calculate it's loss and using the loss we update the weights and biases.

## Calculation

### Overview

Suppose we have to calculate and update the weight $$w^2_{11}$$, we will have to perform the following calculations:

$$
w^2_{11new} = w^2_{11old} - \eta \frac{\partial L}{\partial w^2_{11}}
$$

Now as we already have the value of $$w^2_{11old}$$ & $$\eta$$, we only have to calculate the value of $$\frac{\partial L}{\partial w^2_{11}}$$. This value is also known as the gradient and this is why it's known as that. Now in the algorithm, each weight and bias is updated, so basically in our neural network, we'll calculate such gradient 9 times as we have 6 weights and 3 biases.

### Intuition

Now what does $$\frac{\partial L}{\partial w^2_{11}}$$ actually mean? This term signifies the _amount_ of change occurs in $$L$$ when we change $$w^2_{11}$$.&#x20;

But you might notice, $$L$$ doesn't directly depends upon $$w^2_{11}$$ but rather depends upon $$\hat{y}$$ and then as we already know, $$\hat{y}$$ directly depends upon $$w^2_{11}$$. So basically it means that $$L$$ indirectly depends upon $$w^2_{11}$$ (and that is why we're using $$\partial$$ derivatives and not direct derivatives)

$$
\frac{\partial L}{\partial w^2_{11}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w^2_{11}}
$$

### Raw calculation

$$
\frac{\partial L}{\partial \hat{y}} = \frac{\partial (y - \hat{y})^2}{\partial \hat{y}} = -2(y - \hat{y})
$$

$$
\frac{\partial \hat{y}}{\partial w^2_{11}} = \frac{\partial (O_{11}\cdot w^2_{11} + O_{12}\cdot w^2_{21} + b_{21})}{w^2_{11}}= O_{11}
$$

$$
\frac{\partial L}{\partial w^2_{11}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w^2_{11}}=-2(y-\hat{y})\cdot O_{11}
$$



## Summary

So basically, update all the weights and biases for each row and after we're done with the first iteration, the weights and biases will predict answer a little better than previous turn.

However, it won't be as correct as we want, so for that, we'll let this happen numerous times on our dataset for it to get better each time.

Number of times the algorithm will go through the whole dataset is known as epoch. With each epoch, the loss will be reduced and the model gets better with time.

[^1]: Mean squared error
