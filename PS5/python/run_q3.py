import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 5 ##
learning_rate = 3.5e-3 ## 3.5e-3 best
hidden_size = 64

num_examples = train_x.shape[0]

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
nn_input_size = train_x.shape[1] # unrolled 32x32 image = 1024
nn_output_size = train_y.shape[1] # 36 classes = 26 letters of alphabet + 10 digits
initialize_weights(nn_input_size,hidden_size,params,'layer1')
initialize_weights(hidden_size,nn_output_size,params,'output')
pre_training_weights_layer1 = params['Wlayer1'].copy() # copy for later visualization

# with default settings, you should get loss < 150 and accuracy > 80%
epoch_number = np.arange(max_iters)
epoch_loss = np.zeros(epoch_number.shape)
epoch_train_acc = np.zeros(epoch_number.shape)
epoch_valid_acc = np.zeros(epoch_number.shape)
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)

        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        
        # be sure to add loss and accuracy to epoch totals
        total_loss += loss
        total_acc += acc # sum up now and divide by number of batches at end to get avg.

        # backward
        delta1 = probs
        yb_idx = np.argmax(yb, axis=1)
        delta1[np.arange(probs.shape[0]),yb_idx] -= 1
        
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        params['Woutput'] = params['Woutput'] - learning_rate * params['grad_Woutput']
        params['boutput'] = params['boutput'] - learning_rate * params['grad_boutput']
        
        params['Wlayer1'] = params['Wlayer1'] - learning_rate * params['grad_Wlayer1']
        params['blayer1'] = params['blayer1'] - learning_rate * params['grad_blayer1']
        
    total_loss = total_loss / num_examples # Per FAQ (@562) and @590 on Piazza, losses should be divided by number of samples
    total_acc = total_acc / len(batches)
    
    # Store data for plot:
    epoch_loss[itr] = total_loss # Per FAQ (@562) and @590 on Piazza, losses should be divided by number of samples
    epoch_train_acc[itr] = total_acc
    
    h1 = forward(valid_x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    _, valid_acc = compute_loss_and_acc(valid_y, probs)
    epoch_valid_acc[itr] = valid_acc
        
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

# run on validation set and report accuracy! should be above 75%
h1 = forward(valid_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
_, valid_acc = compute_loss_and_acc(valid_y, probs)

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Produce plots for Q3.1 and Q3.2:
import matplotlib.pyplot as plt
# Accuracy Plot:
plt.figure()
plt.plot(epoch_number, epoch_train_acc)
plt.plot(epoch_number, epoch_valid_acc)
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.title('Q3.1 - Learning Rate: {}, Batch Size: {} \n Final Validation Set Accuracy: {:.4f}'.format(learning_rate, batch_size, valid_acc))
plt.legend(['Accuracy on Training Set', 'Accuracy on Validation Set'])
plt.show()

# Loss Plot:
plt.figure()
plt.plot(epoch_number, epoch_loss)
plt.xlabel('Epoch Number')
plt.ylabel('Cross-Entropy Loss per Sample')
plt.title('Q3.1 - Learning Rate: {}, Batch Size: {}'.format(learning_rate, batch_size, valid_acc))
plt.show()

# Report performance on test set for Q3.2:
h1 = forward(test_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
_, test_acc = compute_loss_and_acc(test_y, probs)
print('Testing accuracy: ',test_acc)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
if True: # True to plot
    # For untrained layer 1:
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols = (8,8), axes_pad=0.04)
    for i in range(pre_training_weights_layer1.shape[1]):
        grid[i].imshow(pre_training_weights_layer1[:,i].reshape(32,32))
    plt.show()
    
    # For trained layer 1:
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols = (8,8), axes_pad=0.04)
    for i in range(params['Wlayer1'].shape[1]):
        grid[i].imshow(params['Wlayer1'][:,i].reshape(32,32))
    plt.show()
    
# Q3.4
confusion_matrix = np.zeros((test_y.shape[1],test_y.shape[1]))

# compute comfusion matrix here
# Run NN forward on testing set (per @563 on Piazza):
h1 = forward(test_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
# Get labels:
prediction = np.argmax(probs, axis=1)
true_class = np.argmax(test_y, axis=1)
# Compute confusion matrix:
# For each true label:
for label in range(test_y.shape[1]):
    # Get predicted values for each with this label:
    preds = prediction[true_class == label]
    # Count occurences of each unique value:
    vals, counts = np.unique(preds, return_counts=True)
    confusion_matrix[label,vals] = counts

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()