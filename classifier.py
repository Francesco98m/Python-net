import numpy as np
import torch

dtype = float
device = torch.device("cpu")

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D, h, K = 32, 2, 100, 3

# Create random input and output data
X = torch.from_numpy(np.load("Data/train_samples" + '.npy'))
y = torch.from_numpy(np.load("Data/train_labels" + '.npy'))

W = 0.01 * torch.randn(D, h, device=device, dtype=dtype, requires_grad=True)
b = torch.zeros(1, h, device=device, dtype=dtype, requires_grad=True)
W2 = 0.01 * torch.randn(h, K, device=device, dtype=dtype, requires_grad=True)
b2 = torch.zeros(1, K, device=device, dtype=dtype, requires_grad=True)

# some hyperparameters
step_size = 1e-0
reg = 1e-3  # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):
    # evaluate class scores, [N x K]
    hidden_layer = (X.mm(W) + b).clamp(0)  # note, ReLU activation
    #  hidden_layer = torch.max(torch.mm(X, W) + b, torch.zeros(b.shape[0])) equivalente a clamp
    scores = hidden_layer.mm(W2) + b2
    # compute the class probabilities
    exp_scores = torch.exp(scores)
    probs = exp_scores / torch.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -torch.log(probs[range(num_examples), y])
    data_loss = torch.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*torch.sum(W.pow(2)) + 0.5*reg*torch.sum(W2.pow(2))
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("iteration %d: loss %f" % (i, loss))
    W.retain_grad()
    W2.retain_grad()
    b.retain_grad()  # ???????????????
    b2.retain_grad()
    loss.backward(retain_graph=True)
    # perform a parameter update
    with torch.no_grad():
        W += -step_size * W.grad
        b += -step_size * b.grad
        W2 += -step_size * W2.grad
        b2 += -step_size * b2.grad
        W.grad.zero_()
        W2.grad.zero_()
        b.grad.zero_()
        b2.grad.zero_()
    """
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = torch.mm(hidden_layer.T, dscores)
    db2 = torch.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = torch.mm(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dW = torch.mm(X.T, dhidden)
    db = torch.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW += reg * W 
    # perform a parameter update  
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2
    """


##test

hidden_layer = (torch.mm(X, W) + b).clamp(0)
scores = torch.mm(hidden_layer, W2) + b2
predicted_class = torch.argmax(scores, axis=1)

np.save("Data/test_scores", predicted_class.numpy())

