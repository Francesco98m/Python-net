import torch
import numpy as np

def SVM_loss(scores, y, delta = 1.0):
    """
    scores is a torch.tensor object of shape (n_categories, n_examples); 
    every column shall contain the score of one example.
    y is a torch.tensor object of shape (n_examples,); it shall contain the correct categories of every example.
    delta is a float number; it's a hyperparameter of this loss function.
    """
    n_examples = scores.shape[1]  
    errors = scores - scores[y,list(range(n_examples))] 
    errors += delta # adding the delta
    errors = errors.clamp(0) # set to zero all elements smaller than zero
    errors[y,list(range(n_examples))]  = 0 # set to zero all elements which were just (scores[i][j] - scores[i][j] + delta)
    loss = (errors.sum())/n_examples

    return loss


def cross_entropy (scores, y):
    """
    scores is a torch.tensor object of shape (n_categories, n_examples);
    every column contains the score of one example.
    y is a torch.tensor object of shape (n_examples,); it shall contain the correct categories of every example.
    The function return the cross entropy loss (already normalized)
    """
    s = torch.gather(scores, 0, y) # we are indexing scores using the value of y
    loss = ((-((s.exp())/(scores.exp().sum())).log())).sum() # computing the loss
    loss = loss / (scores.shape[1]) # normalizing the loss
    return loss


def softmax_function(scores):
    """
    scores is a numpy.tensor of shape (n_categories, n_examples); every column shall contain the 
    output of the net for one example.
    """

    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    return probabilities


def cross_entropy_loss(scores, labels):
    """
    scores is a numpy.tensor of shape (n_categories, n_examples); every column shall contain the 
    output of the net for one example.
    """   
    
    num_examples = scores.shape[0]
    unnormalized_probabilities = torch.exp(scores)
    probs = unnormalized_probabilities / torch.sum(unnormalized_probabilities, axis=1, keepdims= True)   #??????????????????

    correct_logprobs = -torch.log(probs[list(range(num_examples)),labels])
    loss = torch.sum(correct_logprobs)/num_examples

    return loss
   

def regularization_loss(*W):
    """
    Every argument W should be a weight matrix.
    """  
    reg_strenght = 1e-03
    reg_loss = 0
    for A in W:
        reg_loss +=(A.pow(2).sum()).item()
        reg_loss = reg_loss * reg_strenght
        # print(reg_loss.shape)
    return reg_loss
