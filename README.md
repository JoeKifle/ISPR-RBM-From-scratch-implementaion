# RBM (Ristricted Boltzman Machines)


A restricted Boltzmann machine (RBM) is a generative stochastic artificial neural network that can learn a probability distribution over its set of inputs. It is a shallow 2 layer neural network that can be used to find patterns in a data by sampling. In a way that they are similar to Autoencoders. They are mainly used to exctract usefull feature of your input data, like Autoencoders.  They can be visualized as below.

![1_LoeBW9Stm6HjK57yBp45sQ](https://user-images.githubusercontent.com/40062131/119184573-d573ad00-ba75-11eb-9435-b8732a52e756.png)

The hidden unit will be the representation or encoding of our data. The input layer size in case of MNIST dataset is 784 and the size of hidden layer is 100.

The implementation process of RBM can be summerized in to two process. 

<b>1. The forward pass:</b> In this pass every input will be combined with an individual weight and one overall bias. Then the result will be passed to the hidden layer. Then we will apply Gibbs sampling on the hidden layer.<br>

<b>2. The backward pass:</b> Each activated nodes from hidden layer will be used to reconstruct the visible layer and compute our loss.

