from NetworkDefinition import *
from CNNTraining import *
from Network import *
from settings import HYPER_PARAMETERS

#shorthands, makes typing easier
NL = NetworkLayer
AF = ActivationFunction
PI = ParamInitializer
LF = LossFunction
OP = Optimizer

# example of how to make a lenet5 cnn
def LeNet5(num_output_nodes = 10, name = 'LeNet5'):
    # each entry in the list is a layer in the network
    layers = [
        (NL.conv2d, [3, 6, 5]), # the conv2d function (see: nn.Conv2d) takes at least 3 arguments: input size, output size, kernel size
        (NL.avgpool, 2),
        (NL.conv2d, [6, 16, 5]),# this conv2d function has an input size of 6, output size of 16 and kernel size of 5
        (NL.avgpool, 2),
        NL.flatten,             # the flatten layer does not use any arguments. 
        (NL.linear, 400, 120),  # an alias for (NL.linear, [400, 120]), it does the exact same thing
        (NL.linear, 120, 84),   # more arguments can always be given, looking at the documentation of nn.Linear(), it takes 5 arguments, of which the first 2 are mandatory
        (NL.linear, 84, num_output_nodes)
    ]

    # just like the layers, parameters can be passed to each of these functions. I.e. (AF.ReLU, True) to make the be performed inplace
    layer_activation_func = AF.ReLU         # after conv or linear layer, perform a relu activation function
    output_activation_func = AF.SoftMax     # perform a softmax function after the final layer
    initializer = PI.kaiming_uniform        # the initializer function for the weights in the cnn

    return CNN(layers, layer_activation_func, output_activation_func, initializer, name)

def YOLOv1():
    layers = [
        (NL.conv2d, 3, 16, 3, 1, 1),
        (NL.batchNorm, 16),
        (NL.maxpool, 2),
        (NL.conv2d, 16, 32, 3, 1, 1),
        (NL.batchNorm, 32),
        (NL.maxpool, 2),
        (NL.conv2d, 32, 64, 3, 1, 1),
        (NL.batchNorm, 64),
        (NL.maxpool, 2),
        (NL.conv2d, 64, 64, 3, 1, 1),
        (NL.batchNorm, 64),
        (NL.maxpool, 2),
        (NL.conv2d, 64, 32, 3, 1, 1),
        (NL.batchNorm, 32),
        NL.flatten,
        NL.dropout,
        (NL.linear, 224, 512),
        (NL.linear, 512, 343)
    ]

    layer_activation_func = AF.ReLU
    output_activation_func = AF.sigmoid
    initializer = PI.kaiming_uniform

    return CNN(layers, layer_activation_func, output_activation_func, initializer, 'YOLOv1')

# makes a trainer with the given cnn
def MkTrainer(cnn = None):
    learning_rate = HYPER_PARAMETERS
    loss_function = LF.cross_entropy        # just like the layers, parameters can be passed to the loss function and optimizer in the exact same way
    optimizer = (OP.adam, learning_rate)    # in this case, the learning rate is the second argument of optim.Adam(), since the first argument is already fixed
    return TrainingProfile(cnn, loss_function, optimizer)   # see NetworkDefinitions for an overview of what gets translated to what, although the enum names are the same as their pytorch counterparts