# Self-scalable Tanh (Stan) Activation Function
## Applied to Physics-informed Neural Networks (PINN) 
This is the repository for Physics-informed Neural Networks with Stan (Self-scalable Tanh) activation function. 

$Stan(x) = tanh(x) + \beta \times x \times tanh(x)$

$\beta$ is a trainable neuron-wise parameter. The codes are being cleaned for easy usage. Meanwhile, if you already have a PINN code and want to implement the activation function,

In Pytorch, if you have the activation function as $tanh$ you can simply modify it as follows
```
#Initialization
self.beta = Parameter(torch.ones((NN_width,len(layers)-2))) 
self.beta.requiresGrad = True #Add this to the weights and biases initializations
```
```
#In your 'forward pass' function
z = self.activation(x) #tanh 
z = z + self.beta[:,i]* x * z # i denotes the layer index 
```

Please cite if you benefit from this work. [https://ieeexplore.ieee.org/document/10227556]. 
>Note: **DO NOT use the Arxiv preprint version** of the article; there are several inaccuracies.  


Citation (BibTeX):
```
@article{gnanasambandam2023self,
  title={Self-scalable Tanh (Stan): Multi-Scale Solutions for Physics-Informed Neural Networks},
  author={Gnanasambandam, Raghav and Shen, Bo and Chung, Jihoon and Yue, Xubo and Kong, Zhenyu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```

You can also find some other implementations like the [NVIDIA's](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/theory/advanced_schemes.html). I am attaching a figure to show the potential applications of PINN and to grab your attention (of course!) (note: the code for generating this figure is not in the repository).

![](https://github.com/raghavg97/PINN_Stan/blob/main/MP_3D_100resol.gif)


**R. Gnanasambandam, B. Shen, J. Chung, X. Yue and Z. Kong, "Self-scalable Tanh (Stan): Multi-Scale Solutions for Physics-Informed Neural Networks," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2023.3307688.**

