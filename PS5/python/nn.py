import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    # Initialize b:
    b = np.zeros(out_size)
    
    # Initialize W:
    bound = np.sqrt(6 / (in_size + out_size)) # from (16) in paper. Equivalent to sqrt(12*var)/2 = sqrt(3*var) where var = 2/(n_in+n_out).
    W = np.random.uniform(-bound, bound, (in_size, out_size))

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    # max value in float64 is 1.79e+308
    # b/c ln(1.79e+308) = 709.77, x should be clipped at -709, so no number
    # larger than e^(--709) is computed (note: numpy doesn't have a problem with 
    # e^(y) where y is very negative, only when y is very positive)
    xx = np.clip(x, a_min = -709, a_max = None)
    res = 1 / (1 + np.exp(-xx))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = X@W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    row_max = np.max(x, axis=1).reshape((-1,1))
    xx = x - row_max # off set for numerical stablility (never taking exp to a positive power)
    ex = np.exp(xx)
    S = np.sum(ex, axis=1).reshape((-1,1))
    res = ex / S

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    log_probs = np.log(probs)
    # dot multiply each row in y with the corresponding row in log_probs, stack results in a column:
    # example_loss = np.einsum('ij,ij->i', y, log_probs)
    
    loss = -np.sum(y * log_probs)
    
    # highest probability for each row is the prediction:
    prediction = np.argmax(probs, axis=1)
    true_class = np.argmax(y, axis=1)
    
    acc = np.count_nonzero(prediction == true_class) / prediction.shape[0]
    
    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    grad_f = activation_deriv(post_act) * delta

    # Aggregate across all examples
    # (more explicit way of doing it, yields same results, keeping for 
    # personal reference):
    #grad_W = np.zeros(W.shape)
    #grad_b = np.zeros(b.shape)
    #for row in range(X.shape[0]):
        #xx = X[row,:].reshape(1,-1)
        #grad_y = grad_f[row,:].reshape(1,-1)
        
        #grad_W += xx.T @ grad_y
        #grad_b += grad_y.reshape(-1)
    
    # then compute the derivative W,b, and X
    grad_W = X.T @ grad_f
    grad_X = grad_f @ W.T
    grad_b = np.sum(grad_f, axis=0)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    
    current_batch_rows = []
    
    # indices of rows (training examples) in x,y available for the taking:
    available_rows = np.arange(x.shape[0])
    
    # Randomly pick a row (example) from the unchosen rows, add it to the 
    # current batch, and continue until batch is full or we're out of rows (examples):
    while np.size(available_rows) > 0:
        # pick a random row index from available_rows:
        available_rows_index = np.random.randint(0, np.size(available_rows))
        row = available_rows[available_rows_index]
        # add it to the current batch:
        current_batch_rows.append(row)
        # remove it from available_rows:
        available_rows = np.delete(available_rows, available_rows_index)
        # advance the batch if necessary (batch is full or we've run out of rows (examples)):
        if len(current_batch_rows) == batch_size or np.size(available_rows) == 0:
            batch_x = x[current_batch_rows,:]
            batch_y = y[current_batch_rows,:]
            batches.append( (batch_x, batch_y) )
            current_batch_rows = []
    
    return batches
