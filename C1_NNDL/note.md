# Neural Network Basics

1. Logistic Regression as Neural Network

> Logistic Regression can be regard as one of the simplest neural network in natural.
> 
> It serves as a binary classifier

Input $x \in R^{n_x}$ $\rightarrow$ Logistic_NN $\rightarrow$ Output a label y = 0 or 1

Consider m training examples $(x^1,y^1), (x^2, y^2),..., (x^m, y^m)$

$$X = [x^1, x^2, x^3, ..., x^m], \quad x^i \in R^{n_x}, \quad X \in M_{n_x \times m}$$
$$Y = [y^1, y^2, y^3, ..., y^m], \quad y^i \in R^1    , \quad Y \in M_{1 \times m}$$


```
X.shape = (n_x,m)
```

