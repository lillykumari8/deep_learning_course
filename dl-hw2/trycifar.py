from uwnet import *


# def conv_net():
#     l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU),
#             make_maxpool_layer(32, 32, 8, 3, 2),
#             make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
#             make_maxpool_layer(16, 16, 16, 3, 2),
#             make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU),
#             make_maxpool_layer(8, 8, 32, 3, 2),
#             make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU),
#             make_maxpool_layer(4, 4, 64, 3, 2),
#             make_connected_layer(256, 10, SOFTMAX)]
#     return make_net(l)

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU, batchnorm=1),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU, batchnorm=1),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU, batchnorm=1),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU, batchnorm=1),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
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
# train_image_classifier(m, train, batch, iters, rate, momentum, decay)

train_image_classifier(m, train, batch, 2000, rate*10, momentum, decay)
train_image_classifier(m, train, batch, 2000, rate*5, momentum, decay)
train_image_classifier(m, train, batch, 1000, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#

# 8*27*1024 + 16*72*256 + 32*144*64 + 64*288*16 + 256*10
# 1108480
# 221696
# 3072 input
# 72 out

#-------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# 11/15/2018 (using HW0,1 solutions)

# def conv_net():
#     l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU),
#             make_maxpool_layer(32, 32, 8, 3, 2),
#             make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
#             make_maxpool_layer(16, 16, 16, 3, 2),
#             make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU),
#             make_maxpool_layer(8, 8, 32, 3, 2),
#             make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU),
#             make_maxpool_layer(4, 4, 64, 3, 2),
#             make_connected_layer(256, 10, SOFTMAX)]
#     return make_net(l)

# train_image_classifier(m, train, batch, iters, rate, momentum, decay)
# evaluating model...
# ('training accuracy: %f', 0.7165600061416626)
# ('test accuracy:     %f', 0.6639999747276306)

# Using just the convnet with uniform learning rate of 0.01 gives a test accuracy of 66.4%

#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------

# def conv_net():
#     l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU, batchnorm=1),
#             make_maxpool_layer(32, 32, 8, 3, 2),
#             make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU, batchnorm=1),
#             make_maxpool_layer(16, 16, 16, 3, 2),
#             make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU, batchnorm=1),
#             make_maxpool_layer(8, 8, 32, 3, 2),
#             make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU, batchnorm=1),
#             make_maxpool_layer(4, 4, 64, 3, 2),
#             make_connected_layer(256, 10, SOFTMAX)]
#     return make_net(l)

# train_image_classifier(m, train, batch, iters, rate, momentum, decay)
# evaluating model...
# ('training accuracy: %f', 0.780460000038147)
# ('test accuracy:     %f', 0.7153000235557556)

# Using the convnet with batchnorm applied to all except the last layer with uniform learning rate of 0.01 gives a test accuracy of 71.5% which is
# better than the convnet without batch-normalization.

#-------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# def conv_net():
#     l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU, batchnorm=1),
#             make_maxpool_layer(32, 32, 8, 3, 2),
#             make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU, batchnorm=1),
#             make_maxpool_layer(16, 16, 16, 3, 2),
#             make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU, batchnorm=1),
#             make_maxpool_layer(8, 8, 32, 3, 2),
#             make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU, batchnorm=1),
#             make_maxpool_layer(4, 4, 64, 3, 2),
#             make_connected_layer(256, 10, SOFTMAX)]
#     return make_net(l)


# train_image_classifier(m, train, batch, 2000, rate*10, momentum, decay)
# train_image_classifier(m, train, batch, 2000, rate*5, momentum, decay)
# train_image_classifier(m, train, batch, 1000, rate, momentum, decay)

# evaluating model...
# ('training accuracy: %f', 0.7985799908638)
# ('test accuracy:     %f', 0.7294999957084656)

# Using the convnet with batchnorm applied to all except the last layer with step learning rate (0.1 for first 2000 iters, 0.05 for next 2000 iters,
# 0.01 for next 1000 iters) gives a test accuracy of 73%. So, the model using batch-normalization performs better with larger learning rates in the
# beginning & yields overall best performance with test accuracy of 73%.