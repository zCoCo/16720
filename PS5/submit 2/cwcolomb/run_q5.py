import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
nn_input_size = train_x.shape[1] # unrolled 32x32 image = 1024
# Per assignment: 1024to32, 32to32, 32to32, 32to1024
initialize_weights(nn_input_size,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'layer2')
initialize_weights(hidden_size,hidden_size,params,'layer3')
initialize_weights(hidden_size,nn_input_size,params,'output')

batches = get_random_batches(train_x,train_x,batch_size)

# initialize momentum accumulators:
params['m_Wlayer1'] = params['Wlayer1'] * 0.0
params['m_blayer1'] = params['blayer1'] * 0.0

params['m_Wlayer2'] = params['Wlayer2'] * 0.0
params['m_blayer2'] = params['blayer2'] * 0.0

params['m_Wlayer3'] = params['Wlayer3'] * 0.0
params['m_blayer3'] = params['blayer3'] * 0.0

params['m_Woutput'] = params['Woutput'] * 0.0
params['m_boutput'] = params['boutput'] * 0.0

# should look like your previous training loops
epoch_number = np.arange(max_iters)
epoch_loss = np.zeros(epoch_number.shape)
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # forward
        h1 = forward(xb,params,'layer1', relu)
        h2 = forward(h1,params,'layer2', relu)
        h3 = forward(h2,params,'layer3', relu)
        xb_hat = forward(h3,params,'output',sigmoid) # reconstruction of x

        # loss
        diff = (xb_hat - xb)
        diff = np.clip(diff, a_min = 1e-150, a_max = None)
        total_loss += np.sum( diff**2 )

        # backward
        delta1 = 2 * (xb_hat - xb)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)

        # apply gradient
        for name in ['layer1', 'layer2', 'layer3', 'output']:
            params['m_W'+name] = 0.9 * params['m_W'+name] - learning_rate * params['grad_W'+name]
            params['W'+name] = params['W'+name] + params['m_W'+name]
            params['m_b'+name] = 0.9 * params['m_b'+name] - learning_rate * params['grad_b'+name]
            params['b'+name] = params['b'+name] + params['m_b'+name]
            
            params['m_W'+name][np.abs(params['m_W'+name]) < 1e-38] = 0. # round out any insanely small numbers
            params['m_b'+name][np.abs(params['m_b'+name]) < 1e-38] = 0.

    total_loss = total_loss / train_x.shape[0] # Per FAQ (@562) and @590 on Piazza, losses should be divided by number of samples
    epoch_loss[itr] = total_loss

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

import matplotlib.pyplot as plt
plt.figure()
plt.plot(epoch_number, epoch_loss)
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
# run on validation set: 
h1 = forward(valid_x,params,'layer1', relu)
h2 = forward(h1,params,'layer2', relu)
h3 = forward(h2,params,'layer3', relu)
xb_hat = forward(h3,params,'output',sigmoid)

# pull out validation set classes:
valid_classes = valid_data['valid_labels']
# pick 5 at "random" (generated with np randint, use these 5 arbitrary classes for visualization):
classes = np.asarray([24, 18, 19, 26, 22])
# pick 2 images at "random" (use these 2 arbitrary image indices for visualization):
images = np.asarray([71, 27])
for vclass in classes:
    selector = np.nonzero(valid_classes[:, vclass])[0]
    ref_images = valid_x[selector[images],:]
    recon_images = xb_hat[selector[images],:]
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(ref_images[0,:].reshape(32,32).T, cmap=plt.cm.gray)
    plt.title('True Image {} \n in Class {}'.format(images[0], vclass))
    
    plt.subplot(222)
    plt.imshow(recon_images[0,:].reshape(32,32).T, cmap=plt.cm.gray)
    plt.title('Reconstructed Image {} \n in Class {}'.format(images[0], vclass))
    
    
    plt.subplot(223)
    plt.imshow(ref_images[1,:].reshape(32,32).T, cmap=plt.cm.gray)
    plt.title('True Image {} \n in Class {}'.format(images[1], vclass))
    
    plt.subplot(224)
    plt.imshow(recon_images[1,:].reshape(32,32).T, cmap=plt.cm.gray)
    plt.title('Reconstructed Image {} \n in Class {}'.format(images[1], vclass))    
    plt.tight_layout()
    plt.show()


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
psnr_avg = 0
for im in range(valid_x.shape[0]):
    psnr_avg += psnr(valid_x[im,:], xb_hat[im,:]) # sum up first then avg out
psnr_avg = psnr_avg / valid_x.shape[0]

print("Average PSNR across Validation Set: {}".format(psnr_avg))
    
