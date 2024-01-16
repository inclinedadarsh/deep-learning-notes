# How does it work?

In this part, we'll write our own back propagation algorithm to train 2 different neural networks or 2 different datasets.

The video I followed while making this part is here: [https://youtu.be/ma6hWrU-LaI?si=cgahIMbJrmiUl505](https://youtu.be/ma6hWrU-LaI?si=cgahIMbJrmiUl505)

## Regression problem

### Dataset & neural network

We'll consider the following dataset and we'll train a model by writing our own algorithm.

| CGPA | Resume score | Package |
| ---- | ------------ | ------- |
| 8    | 8            | 4       |
| 7    | 9            | 5       |
| 6    | 10           | 6       |
| 5    | 12           | 7       |

Below is our simple neural network and we'll be training that only.

<img src="../../.gitbook/assets/file.excalidraw (3).svg" alt="Our neural network for this case" class="gitbook-drawing">

### Algorithm

1. for i in range(epochs):
   1. for j in range(X.shape\[0]):
      1. Select random row
      2. Predict using forward propagation
      3. Calculate loss (loss function is MSE[^1])
      4. Update weights and biases using GD\

2. Calculate average loss for each epoch

### Formula

Now in the last part we derived the formula for finding each weight and bias

$$
w_{n} = w_{o}  -  \eta\frac{{\partial L}}{{\partial w_{o}}}
$$

According to the [above neural network](how-does-it-work.md#dataset-and-neural-network), here are all the formulas:

$$
\frac{\partial L}{\partial w^2_{11}}=-2(y-\hat{y})\cdot O_{11}
$$

$$
\frac{\partial L}{\partial w^2_{21}}=-2(y-\hat{y})\cdot O_{12}
$$

$$
\frac{\partial L}{\partial b_{21}}=-2(y-\hat{y})
$$

$$
\frac{\partial L}{\partial w^1_{11}}=-2(y-\hat{y})\cdot w^2_{11} \cdot x_{11}
$$

$$
\frac{\partial L}{\partial w^1_{21}}=-2(y-\hat{y})\cdot w^2_{11} \cdot x_{12}
$$

$$
\frac{\partial L}{\partial b_{11}}=-2(y-\hat{y})\cdot w^2_{11}
$$

$$
\frac{\partial L}{\partial w^1_{12}}=-2(y-\hat{y})\cdot w^2_{21} \cdot x_{11}
$$

$$
\frac{\partial L}{\partial w^1_{22}}=-2(y-\hat{y})\cdot w^2_{21} \cdot x_{12}
$$

$$
\frac{\partial L}{\partial b_{12}}=-2(y-\hat{y})\cdot w^2_{21}
$$

Ufff... there were quite some formulas here, but it's important to understand how they were formed. If you believe you already know how they were derived, feel free to skip it. <mark style="color:yellow;">You don't need to learn it!</mark>

### Implementation

Now we'll implement this in jupyter notebook. I have done this in a google colab and here's the link of the colab:&#x20;

{% embed url="https://colab.research.google.com/drive/1auLYprIESEktck6RmLkI645FgJCWLYr_?usp=sharing" %}
Google colab for regression problem
{% endembed %}

As you can see, we have implemented an algorithm for traning deep learning neural network in python itself. Let's move on to the classification problem.

***

## Classification problem

### Dataset and neural network

For classification problem we're going to consider a similar dataset.

<table><thead><tr><th data-type="number">CGPA</th><th data-type="number">Resume Score</th><th data-type="number">Is palced?</th></tr></thead><tbody><tr><td>8</td><td>8</td><td>1</td></tr><tr><td>7</td><td>9</td><td>1</td></tr><tr><td>6</td><td>10</td><td>0</td></tr><tr><td>5</td><td>5</td><td>0</td></tr></tbody></table>

<img src="../../.gitbook/assets/file.excalidraw (3).svg" alt="Our neural network for this case" class="gitbook-drawing">

But this time, there are two differences:

1. The activation function we're going to use is [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid\_function)
2. Rather than using the MSE loss function, this time we're going to use [Binary Cross Entropy](#user-content-fn-2)[^2]

$$
LOSS = -y\cdot\log{(\hat{y})}-(1-y)\cdot \log{(1-\hat{y})}
$$

### Algorithm

The algorithm for back propagation in classification problem is same as [#algorithm](how-does-it-work.md#algorithm "mention") in [#regression-problem](how-does-it-work.md#regression-problem "mention")

### Formula

Now for the regression problem, we already derived it's formula in the [what-is-back-propagation.md](what-is-back-propagation.md "mention"), but we didn't do that for classification problem.

However, you can follow similar approach to derive all the formulas for classification problem using the Binary Cross Entropy loss function.

If you want to see how these are dervied, I stronly suggest you checking out [the video](https://youtu.be/ma6hWrU-LaI?si=5R0Waf7FR5ZveHrd\&t=1990).

For now, I'll simply note them down:

$$
\frac{\partial L}{\partial w^2_{11}} = -(y-\hat{y})\cdot O_{11}
$$

$$
\frac{\partial L}{\partial w^2_{21}} = -(y-\hat{y})\cdot O_{12}
$$

$$
\frac{\partial L}{\partial b_{21}} = -(y-\hat{y})
$$

$$
\frac{\partial L}{\partial w^2_{11}} = -(y-\hat{y})\cdot w^2_{11}\cdot O_{11}\cdot(1-O_{11})\cdot x_{i1}
$$

$$
\frac{\partial L}{\partial w^2_{21}} = -(y-\hat{y})\cdot w^2_{11}\cdot O_{11}\cdot(1-O_{11})\cdot x_{i2}
$$

$$
\frac{\partial L}{\partial b_{11}} = -(y-\hat{y})\cdot w^2_{11}\cdot O_{11}\cdot(1-O_{11})
$$

$$
\frac{\partial L}{\partial w^1_{12}} = -(y-\hat{y})\cdot w^2_{21}\cdot O_{12}\cdot(1-O_{12})\cdot x_{i1}
$$

$$
\frac{\partial L}{\partial w^1_{22}} = -(y-\hat{y})\cdot w^2_{21}\cdot O_{12}\cdot(1-O_{12})\cdot x_{i2}
$$

$$
\frac{\partial L}{\partial w^2_{12}} = -(y-\hat{y})\cdot w^2_{21}\cdot O_{12}\cdot(1-O_{12})
$$



### Implementation

I have done the implementation in another google colab. Here's the link:

{% embed url="https://colab.research.google.com/drive/10Z8KeWVAQasB6n_MdeZYa07zeNoFTa2G?usp=sharing" %}
Google colab for classification problem
{% endembed %}

You might think that we didn't converge[^3], but we got similar results as Keras. Therefore we can conclude that we have successfully implemented backpropagation for a classification problem with Cross Binary Entropy as our loss function.

***

That's it for back propagation algorithm. We have seen back propagation in detail. Now if you have to developer a better understanding (intuitively) about back-propagation, I recommend you watching CampusX's "[The Why of Back propagation](https://youtu.be/6xO-x8y0YSY?si=nxwYCp1qL96PVRLu)" video.

Otherwise, if you feel confident in this topic, feel free to move on to the next one. Thanks a lot for reading! I'll really appreciate if you have any feedback.

Byee folks :wave:

[^1]: $$(y - \hat{y})^2$$

[^2]: $$LOSS = y\cdot \log{(\hat{y})}-(1-y)\cdot \log{(1 - \hat{y})}$$

[^3]: Didn't minimize loss
