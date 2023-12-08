This is the repository for Physics-informed Neural Networks with Stan (Self-scalable Tanh) activation function. 

$Stan(x) = tanh(x) + \beta x tanh(x)$

$\beta$ is a trainable neuron-wise parameter. The codes are being cleaned for easy usage. Meanwhile, if you already have a PINN code and want to implement the activation function,

In Pytorch, if you have the activation function as $tanh$ you can simply modify it as follows
```
self.beta = Parameter(torch.ones((NN_width,len(layers)-2))) 
self.beta.requiresGrad = True #Add this to the weights and biases initializations
```
```
z = self.activation(x) #tanh 
z = z + self.beta[:,i]* x * self.activation(x) # i denotes the layer index 
```

Please cite if you benefit from this work. [https://ieeexplore.ieee.org/document/10227556]. Note: DO NOT use the Arxiv preprint version of the article, there are several inaccuracies.  

R. Gnanasambandam, B. Shen, J. Chung, X. Yue and Z. Kong, "Self-scalable Tanh (Stan): Multi-Scale Solutions for Physics-Informed Neural Networks," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2023.3307688.
