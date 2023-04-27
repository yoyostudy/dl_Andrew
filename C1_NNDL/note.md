# Neural Network Basics

1. Logistic Regression as Neural Network

    > Logistic Regression can be regard as one of the simplest neural network in natural.
    > 
    > It serves as a binary classifier

    Input $x \in R^{n_x}$ $\rightarrow$ Logistic_NN $\rightarrow$ Output a label y = 0 or 1

    Consider m training examples $(x^1,y^1), (x^2, y^2),..., (x^m, y^m)$

    - $X = [x^1, x^2, x^3, ..., x^m], \quad x^i \in R^{n_x}, \quad X \in M_{n_x \times m}$
    - $Y = [y^1, y^2, y^3, ..., y^m], \quad y^i \in R^1    , \quad Y \in M_{1 \times m}$

      ```
      X.shape = (n_x,m)
      Y.shape = (1,m)
      ```
    - Parameters: $w \in R^{n_x}, b \in R$
    - Prediction: 
        - $\hat{y} = \sigma(w^t x + b) = \sigma (z), \ where \ z := w^t x + b, \ \sigma = (1+e^{-z})^{-1}$
        - $\hat{y^{\color{green} i}} = \sigma(w^t x^{\color{green} i} + b) = \sigma (z^{\color{green} i}) \in R^1, \ {\color{green} i} = 1,2,..m$
        - $Z = [ w^t x^{\color{green} 1} + b,  w^t x^{\color{green} 2} + b, ...,  w^t x^{\color{green} m} + b] = w^t [x^{\color{green} 1},...x^{\color{green} m}] + [b, b, ..., b] \in R^{1 \times n_x \times n_x \times m } = R^{1 \times m }$
        - $Y_{hat} = [\hat{y^{\color{green} 1}}, \hat{y^{\color{green} 2}} , ..., \hat{y^{\color{green} m}}] \in R^{1 \times m}$
        ```
        W.shape = (n_x, 1)
        b.shape = (1,1)
        Z = np.dot(w.T, X) + b
        Z.shape = (1,m)
        Y_hat = sigmoid(Z)
        Y_hat.shape(1,m)
        ```
    - Loss: $L(\hat{y},y) =  - y log \hat{y} - (1-y) log (1-\hat{y}) \large $
    - Cost: $J(w,b) := \large \frac{\sum_{i=1...m}  L(\hat{y^i},y^i)}{m} $
    - Gradient: 
        
        Using Chain's Rule:
        - $\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1...m} \frac{\partial  L(\hat{y^i},y^i)}{\partial b}  =  \frac{1}{m} \sum_{i=1...m} \frac{\partial  L(\hat{y^i},y^i)}{\partial \hat{y^i} } \frac{\partial \hat{y^i}}{\partial b}  = \frac{1}{m} \sum_{i=1...m} \frac{\partial  L(\hat{y^i},y^i)}{\partial \hat{y^i} } \frac{d \hat{y^i}}{d z^i} \frac{\partial z^i}{\partial b} $
        - $\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1...m} \frac{\partial  L(\hat{y^i},y^i)}{\partial w_j}  =  \frac{1}{m} \sum_{i=1...m} \frac{\partial  L(\hat{y^i},y^i)}{\partial \hat{y^i} } \frac{\partial \hat{y^i}}{\partial w_j}  = \frac{1}{m} \sum_{i=1...m} \frac{\partial  L(\hat{y^i},y^i)}{\partial \hat{y^i} } \frac{d \hat{y^i}}{d z^i} \frac{\partial z^i}{\partial w_j}, \ where \  j = 1,2,..., n_x $
        
        Now let us calculate those derivatives and paritial derivatives resepctively:
        - Note that $z^i = \sum_{j = 1...m} w_j x_j^i + b \Rightarrow \frac{\partial z^i}{\partial w_j} = x_j^i , \  \frac{\partial z^i}{\partial b}=1$
        - $\frac{d \hat{y^i}}{d z^i} = \sigma'(z) = (1+e^{-z})^{-2} e^{-z} = \hat{y^i} - \hat{y^i}^2 $ 
        - $\frac{\partial  L(\hat{y^i},y^i)}{\partial \hat{y^i} } = -\frac{y}{\hat{y}} + (1-y) \frac{1-y}{1-\hat{y}}$ 

        Take in all those intermediate derivatives:
        - _differential_ $db = \frac{\partial J}{\partial b} =  \frac{1}{m} \sum_{i=1...m} (\hat{y^i} -y^i) $ 
        - _differential_ $dw_j = \frac{\partial J}{\partial w_j} =   \frac{1}{m} \sum_{i=1...m} (\hat{y^i} -y^i) x_j^i $
        
        ```
        db = np.sum(Y_hat-Y)
        db /= m
        dW = np.dot(X, (Y_hat-Y).T)
        dW /= m
        ```

    - Gradient Descent:
        - $w_j \leftarrow w_j - \alpha dw_j , \ j = 1,..., n_x$
        - $b   \leftarrow b - \alpha db$
      
    
    
    



