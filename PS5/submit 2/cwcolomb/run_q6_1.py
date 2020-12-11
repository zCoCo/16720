import torch
import scipy.io
from nn import *
from collections import Counter

## IMPORT DATA:
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = torch.from_numpy(valid_data['valid_data']), torch.from_numpy(valid_data['valid_labels'])
test_x, test_y = torch.from_numpy(test_data['test_data']), torch.from_numpy(test_data['test_labels'])

## SETTINGS:
max_iters = 50
# pick a batch size, learning rate
batch_size = 5 ##
learning_rate = 3.5e-3 ## 3.5e-3 best
hidden_size = 64


## SETUP:
num_examples = train_x.shape[0]

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
nn_input_size = train_x.shape[1] # unrolled 32x32 image = 1024
nn_output_size = train_y.shape[1] # 36 classes = 26 letters of alphabet + 10 digits
w1 = torch.randn(nn_input_size,hidden_size, requires_grad=True)
b1 = torch.randn(hidden_size, requires_grad=True)
w2 = torch.randn(hidden_size, nn_output_size, requires_grad=True)
b2 = torch.randn(nn_output_size, requires_grad=True)

model= torch.nn.Sequential(
    torch.nn.Linear(nn_input_size,hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size,nn_output_size),
    torch.nn.Softmax()
    )

## TRAIN:
epoch_number = np.arange(max_iters)
epoch_loss = np.zeros(epoch_number.shape)
epoch_train_acc = np.zeros(epoch_number.shape)
epoch_valid_acc = np.zeros(epoch_number.shape)
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        xb = torch.from_numpy(xb)
        yb = torch.from_numpy(yb)
        
        # forward
        probs = model(xb.float())

        # loss and accuracy:
        true_class = torch.argmax(yb, axis=1)
        loss = torch.nn.functional.cross_entropy(probs, true_class)
        _, acc = compute_loss_and_acc(yb.detach().numpy(), probs.detach().numpy())
        
        # be sure to add loss and accuracy to epoch totals
        total_loss += loss.data.detach().numpy()
        total_acc += acc # sum up now and divide by number of batches at end to get avg.

        # backward
        loss.backward()
        
        # apply gradient
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
        model.zero_grad()
        
    total_loss = total_loss / num_examples # Per FAQ (@562) and @590 on Piazza, losses should be divided by number of samples
    total_acc = total_acc / len(batches)
    
    # Store data for plot:
    epoch_loss[itr] = total_loss # Per FAQ (@562) and @590 on Piazza, losses should be divided by number of samples
    epoch_train_acc[itr] = total_acc
        
    probs = model(valid_x.float())
    _, valid_acc = compute_loss_and_acc(valid_y.detach().numpy(), probs.detach().numpy())
    epoch_valid_acc[itr] = valid_acc
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))


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
