from uwnet import *

def softmax_model():
    l = [make_connected_layer(3072, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(3072, 32, LRELU),
            make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)

def conv_net():
    # How many operations are needed for a forard pass through this network?
    # Your answer: 1108480
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)


# def modified_conv_net():
    
#     l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, RELU),
#             make_maxpool_layer(32, 32, 8, 3, 2),
#             make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
#             make_convolutional_layer(16, 16, 16, 16, 3, 2, LRELU),
#             make_connected_layer(8*8*16, 256, LRELU),
#             make_connected_layer(256, 10, SOFTMAX)]
#     return make_net(l)


def your_net():
    # Define your network architecture here. It should have 5 layers. How many operations does it need for a forward pass? 970560
    # It doesn't have to be exactly the same as conv_net but it should be close.
    l = [make_connected_layer(3072, 300, LRELU),
        make_connected_layer(300, 128, LRELU),
        make_connected_layer(128, 64, LRELU),
        make_connected_layer(64, 32, LRELU),
        make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)



print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test", "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
# train_image_classifier(m, train, batch, iters/2, rate*.1, momentum, decay)

print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:


#convnet

# evaluating model...
# ('training accuracy: %f', 0.6954200267791748)
# ('test accuracy:     %f', 0.6485999822616577)




# def modified_conv_net():
#     l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, RELU),
#             make_maxpool_layer(32, 32, 8, 3, 2),
#             make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
#             make_convolutional_layer(16, 16, 16, 16, 3, 2, LRELU),
#             make_connected_layer(8*8*16, 256, LRELU),
#             make_connected_layer(256, 10, SOFTMAX)]
#     return make_net(l)

# evaluating model...
# ('training accuracy: %f', 0.8389599919319153)
# ('test accuracy:     %f', 0.6428999900817871)


# def your_net():
#     l = [make_connected_layer(3072, 300, LRELU),
#         make_connected_layer(300, 128, LRELU),
#         make_connected_layer(128, 64, LRELU),
#         make_connected_layer(64, 32, LRELU),
#         make_connected_layer(32, 10, SOFTMAX)]
#     return make_net(l)


# evaluating model...
# ('training accuracy: %f', 0.5516999959945679)
# ('test accuracy:     %f', 0.5101000070571899)

#--------------------------------------------------------------------------------------------------------------------------------------------


# The provided model of conv_net is able to achieve test accuracy of 65%.
# The number of operations involved in the forward pass for a convolutional layer l is equal to l.filters * (l.size * l.size * l.channels) *(out_w*out_h) where out_w = (l.w - 1)/l.stride -1, similarily out_h
# Hence, the number of operations involved in the forward pass for the given conv_net are as follows:
# 1) conv layer1 = 8 * (3*3*3) * (32*32) = 221184
# 2) conv layer2 = 16 * (3*3*8) * (16*16) = 294912
# 3) conv layer3 =  32 * (3*3*16) * (8*8) = 294912
# 4) conv layer4 = 64 * (3*3*32) * (4*4) = 294912
# 5) fc layer = 256 * 10 = 2560

# Overall = 1108480

# The fully connected network that I have defined has 5 layers as following and achieves test accuracy of 51%:
#         make_connected_layer(3072, 300, LRELU)
#         make_connected_layer(300, 128, LRELU)
#         make_connected_layer(128, 64, LRELU)
#         make_connected_layer(64, 32, LRELU)
#         make_connected_layer(32, 10, SOFTMAX)

# So, the number of operations involved in this network are as follows:
# 1) layer1 = 3072*300 = 921600
# 2) layer2 = 300*128 = 38400
# 3) layer3 = 128*64 = 8192
# 4) layer4 = 64*32 = 2048
# 5) layer5 = 32*10 = 320

# Overall = 970560

# The conv_net performs better than the fully connected network (your_net) since convolutional layers followed by max-pooling are able to better
# extract the spatial information and hence extract better features wrt image data (spatial local + global features) while fully connected layers 
# generally cause loss of spatial information since each neuron in input layer is connected to each output layer neuron.