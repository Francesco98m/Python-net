import torch

def SVM_loss_linear(X, y, W, delta = 1.0, regularization = False):
    """
    X is a torch.tensor object of shape (n_inputs, n_examples); every column of X shall contain one example.
    y is a torch.tensor object of shape (n_examples,); it shall contain the correct categories of every example.
    W is torch.tensor object of shape (n_categories, n_inputs).
    delta is a float number; it's a hyperparameter of this loss function.
    """
    try:
        n_examples = X.shape(1)   
        scores = torch.mm(W,X)    #(n_categories)x(n_examples), every column represents the score of one example in X
        errors = scores - scores[[[y],[list(range(n_examples))]]] 
# Happens the following: 
#scores is a tensor of shape (n_categories, n_examples) and the second term is a tensor of shape (n_examples,); subtracting the latter to the first you will subtract to all the elements of the first column of scores the first element of the second term, and so on. 
#So from scores we must choose for every column (every score) the element which represent the correct categorie (for every column the position of the corrent element is contained in y). In order to do that we write:
# scores[[[y],[list(range(n_examples))]]] 
# which is equal to:
# scores[[[correct_position_1, correct_position_2, ..., correct_position_nexamples],[0,1,2,...,n_examples]]] 
# which in turn is equal to what we wanted:
# [scores[correct_position_1][0]  ,  scores[correct_position_2,1]  ,   scores[correct_position_3, 2] , ... ]
        errors += delta # adding the delta
        errors = errors.clamp(0) # set to zero all elements smaller than zero
        errors[[[y][list(range(n_examples))]]] = 0 # set to zero all elements which were just (scores[i][j] - scores[i][j] + delta)
        loss = (errors.sum())/n_examples
    
        if (regularization):
            regularization_loss = W.pow(2).sum()
            loss += regularization_loss
    
        return loss
    except:
        print("Error number +39 329 720 4110\n")
        return 0     
    
    
    
    
