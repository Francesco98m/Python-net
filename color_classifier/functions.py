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
# Happens the following: 
#scores is a tensor of shape (n_categories, n_examples) and the second term is a tensor of shape (n_examples,); 
#subtracting the latter to the first you will subtract to all the elements of the first column of scores the first element 
#of the second term, and so on. 
#So from scores we must choose for every column (every score) the element which represent the correct categorie (for every 
#column the position of the corrent element is contained in y). In order to do that we write:
# scores[y,list(range(n_examples))] 
# which is equal to:
# scores[[correct_position_1, correct_position_2, ..., correct_position_nexamples],[0,1,2,...,n_examples]] 
# which in turn is equal to what we wanted:
# [scores[correct_position_1][0]  ,  scores[correct_position_2,1]  ,   scores[correct_position_3, 2] , ... ]
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
    
    num_examples = scores.shape[1]
    unnormalized_probabilities = torch.exp(scores)
    probs = unnormalized_probabilities / torch.sum(unnormalized_probabilities, axis=0, keepdims=True)

    correct_logprobs = -torch.log(probs[labels, range(num_examples)])
    loss = torch.sum(correct_logprobs)/num_examples

    return loss
   

def regularization_loss(*W):
    """
    Every argument W should be a weight matrix.
    """  
    reg_strenght = 1e-03
    reg_loss = 0
    for A in W:
        reg_loss = reg_loss + (A.pow(2).sum()).item()
        reg_loss = reg_loss * reg_strenght
        # print(reg_loss.shape)
    return reg_loss