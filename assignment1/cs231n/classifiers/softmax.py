import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_data = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  # subtracting the maximum score so we can stabilize the outcomes and avoid any calculation errors we might get
  stabilized_scores = scores - np.amax(scores,axis=1,keepdims=True)
  for i in range(num_data):
        curr_stabilized_scores = stabilized_scores[i,:]
        loss += -(curr_stabilized_scores[y[i]]) + np.log(np.sum(np.exp(curr_stabilized_scores)))
        for j in range(num_class):
            # took the derivative of softmax function w.r.t W for each Wj
            delta_score = np.exp(curr_stabilized_scores[j])/np.sum(np.exp(curr_stabilized_scores))
            if j == y[i]:
                dW[:,j] += (-1 + delta_score)*X[i]
            else:
                dW[:,j] += delta_score*X[i]
  
  loss /= num_data
  loss += np.sum(W*W)*reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW /= num_data
  dW += W*2*reg
    
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0   
  dW = np.zeros_like(W)
  scores = X.dot(W)
  num_data = X.shape[0]
  num_class = W.shape[1]  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  stabilized_scores = scores - np.amax(scores,axis=1,keepdims=True)
  true_class_scores = stabilized_scores[np.arange(stabilized_scores.shape[0]),y]
  true_class_scores.shape = (stabilized_scores.shape[0],1)
  loss_matrix = -true_class_scores + np.log(np.sum(np.exp(stabilized_scores),axis=1,keepdims=True))
  loss = np.sum(loss_matrix)
  loss /= num_data
  loss += reg*np.sum(W*W)  
  
  delta_scores =  np.exp(stabilized_scores)/ np.sum(np.exp(stabilized_scores), axis=1,keepdims=True)
  delta_scores[np.arange(num_data),y] = delta_scores[np.arange(num_data),y] - 1
  
  dW = np.dot(X.T,delta_scores)
  dW /= num_data
  dW += reg*W*2
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

