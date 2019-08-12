## 1 dimensional input
Input  
$$
    \bm{X} = \left(
        \begin{array}{c}
            x_0 \\
            x_1 \\
            \vdots \\
            x_{N-1}
        \end{array}
    \right)
$$
Label vector
$$
    \bm{T} = \left(
        \begin{array}{c}
            t_0 \\
            t_1 \\
            \vdots \\
            t_{N-1}
        \end{array}
    \right)
$$

conditional probability  
$$
    P(t=1|x)
$$  

simple likelihood model  
$$
    P(t=1|x)=w
$$  
$$
    P(\bm{T}=0,0,0,1|x)=(1-w^3)w
$$  

log likelihood  
$$
    log P = log \{(1-w)^3 w\}=3log(1-w) + logw
$$  
$$
    \frac{\partial}{\partial w} log P = \frac{\partial}{\partial w}[3log(1-w) + logw] = 0
$$  
$$
    3 \frac{-1}{1-w} + \frac{1}{w} = 0
$$  
$$
    w = \frac{1}{4}
$$  

logistic regression  
$$
    y = w_0 x + w_1
$$  
$$
    y = \sigma (w_0 x + w_1) = \frac{1}{1 + exp\{-(w_0 x + w_1)\}}
$$  