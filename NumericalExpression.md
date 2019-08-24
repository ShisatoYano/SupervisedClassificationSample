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
$$
    y = \sigma (w_0 x + w_1) = P(t=1|x)
$$  
$$
    P(t|x) = y^t (1-y)^{1-t}
$$  
$$
    P(\bm{T}|\bm{X}) = \prod_{n=0}^{N-1} P(t_n|x_n) = \prod_{n=0}^{N-1} {y_n}^{t_n} (1-y_n)^{1-t_n}
$$  
$$
    log P(\bm{T}|\bm{X}) = \sum_{n=0}^{N-1} \{ t_n log y_n + (1-t_n) log (1-y_n) \}
$$  

cross entropy error function  
$$
    E(\bm{W}) = -\frac {1}{N} log P(\bm{T}|\bm{X}) = -\frac {1}{N} \sum_{n=0}^{N-1} \{ t_n log y_n + (1-t_n) log (1-y_n) \}
$$  

Gradient method of cross entropy error  
$$
    \frac{\partial}{\partial w_0}E(\bm{W}) = \frac{1}{N}\frac{\partial}{\partial w_0}\sum_{n=0}^{N-1}E_n(\bm{W}) = \frac{1}{N}\sum_{n=0}^{N-1}\frac{\partial}{\partial w_0}E_n(\bm{W})
$$  
$$
    y_n = \sigma(a_n) = \frac{1}{1 + exp(-a_n)}
$$  
$$
    a_n = w_0 x_n + w_1
$$
$$
    \frac{\partial E_n}{\partial w_0} = \frac{\partial E_n}{\partial y_n} \frac{\partial y_n}{\partial a_n} \frac{\partial a_n}{\partial w_0}
$$  
$$
    \frac{\partial E_n}{\partial y_n} = -t_n\frac{\partial}{\partial y_n}logy_n - (1-t_n)\frac{\partial}{\partial y_n}log(1-y_n) = - \frac{t_n}{y_n} + \frac{1-t_n}{1-y_n}
$$  
$$
    \frac{\partial y_n}{\partial a_n} = \frac{\partial}{\partial a_n}\sigma(a_n)=\sigma(a_n)\{1-\sigma(a_n)\} = y_n(1 - y_n)
$$  
$$
    \frac{\partial a_n}{\partial w_0} = \frac{\partial}{\partial w_0}(w_0 x_n + w_1) = x_n
$$  
$$
    \frac{\partial E_n}{\partial w_0} = (-\frac{t_n}{y_n} + \frac{1-t_n}{1-y_n})y_n(1-y_n)x_n = (y_n - t_n)x_n
$$  
$$
    \frac{\partial E}{\partial w_0} = \frac{1}{N}\sum_{n=0}^{N-1}(y_n-t_n)x_n
$$
$$
    \frac{\partial E}{\partial w_1} = \frac{1}{N}\sum_{n=0}^{N-1}(y_n-t_n)
$$  

logistic regression on 2 dimension  
$$
    y = \sigma (a),a = w_0 x_0 + w_1 x_1 + w_2
$$  
$$
    \frac{\partial E}{\partial w_0} = \frac{1}{N}\sum_{n=0}^{N-1}(y_n-t_n)x_{n0}
$$  
$$
    \frac{\partial E}{\partial w_1} = \frac{1}{N}\sum_{n=0}^{N-1}(y_n-t_n)x_{n1}
$$
$$
    \frac{\partial E}{\partial w_2} = \frac{1}{N}\sum_{n=0}^{N-1}(y_n-t_n)
$$

logistic regression for 3 classes classification  
$$
    a_k = w_{k0} x_0 + w_{k1} x_1 + w_{k2} (k=0,1,2)
$$  
$$
    a_k = w_{k0} x_0 + w_{k1} x_1 + w_{k2} x_2 = \sum_{i=0}^{D}w_{ki}x_i
$$  
$$
    u = exp(a_0) + exp(a_1) + exp(a_2) = \sum_{k=0}^{K-1}exp(a_k)
$$  
$$
    y_k = \frac{exp(a_k)}{u}
$$  
$$
    \bm{W} = \left[
    \begin{array}{ccc}
      W_{00} & W_{10} & W_{20} \\
      W_{01} & W_{11} & W_{21} \\
      W_{02} & W_{12} & W_{22}
    \end{array}
  \right]
$$  
$$
    P(\bm{t}=[1,0,0]|\bm{x})
$$  
$$
    P(\bm{t}=[0,1,0]|\bm{x})
$$  
$$
    P(\bm{t}=[0,0,1]|\bm{x})
$$  
$$
    P(\bm{t}|\bm{x})=y_0^{t_0}y_1^{t_1}y_2^{t_2}
$$  
$$
    P(\bm{T}|\bm{X}) = \prod_{n=0}^{N-1} P(t_n|x_n) = \prod_{n=0}^{N-1} {y_{n0}}^{t_{n0}} {y_{n1}}^{t_{n1}} {y_{n2}}^{t_{n2}} = \prod_{n=0}^{N-1} \prod_{k=0}^{K-1} {y_{nk}}^{t_{nk}}
$$  
$$
    E(\bm{W}) = -\frac {1}{N} log P(\bm{T}|\bm{X}) = -\frac {1}{N} \prod_{n=0}^{N-1} log P(t_n|x_n) = -\frac {1}{N} \sum_{n=0}^{N-1} \sum_{k=0}^{K-1} t_{nk} log y_{nk}
$$  
$$
    \frac{\partial E}{\partial w_{ki}} = \frac{1}{N}\sum_{n=0}^{N-1}(y_{nk}-t_{nk})x_{ni}
$$  