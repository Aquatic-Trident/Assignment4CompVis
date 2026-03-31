import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from enum import Enum

# this class contains definitions around which cnns and their trainers can be defined

__all__ = ['ActivationFunction', 
           'NetworkLayer', 
           'ParamInitializer', 
           'LossFunction', 
           'Optimizer', 
           'CreateNetworkLayer',
           'CreateActivationFuncion',
           'CreateParamInitializer',
           'CreateLossFunction',
           'CreateOptimizer',
           'Unpack',
           'ActivationLayers'
           ]

#note: if you add elements to these enums, don't forget to update their respective create functions
NetworkLayer = Enum('NetworkLayer', [('conv2d', 1), ('avgpool', 2), ('maxpool', 3), ('flatten', 4), ('linear', 5), ('dropout', 6), ('batchNorm', 7)]) 
ActivationFunction = Enum('ActivationFunction', [('ReLU', 1), ('LeakyReLU', 2), ('SoftMax', 3), ('sigmoid', 4)])
ParamInitializer = Enum('ParamInitializer', [('kaiming_uniform', 1)])
LossFunction = Enum('LossFunction', [('cross_entropy', 1)])
Optimizer = Enum('Optimizer', [('adam', 1)])
ActivationLayers = [NetworkLayer.conv2d, NetworkLayer.linear] # every layer that has an activation function at the end

# below are the create functions that turn the respective enums and parameters into functions for the cnn
def CreateNetworkLayer(layer_info):
    layer, params = Unpack(layer_info)
    match layer:
        case NetworkLayer.conv2d:
            return nn.Conv2d(*params)
        case NetworkLayer.avgpool:
            return nn.AvgPool2d(*params)
        case NetworkLayer.maxpool:
            return nn.MaxPool2d(*params)
        case NetworkLayer.flatten:
            return lambda x: torch.flatten(x, 1, *params)
        case NetworkLayer.linear:
            return nn.Linear(*params)
        case NetworkLayer.dropout:
            return nn.Dropout2d(*params)
        case NetworkLayer.batchNorm:
            return nn.BatchNorm2d(*params)
        
def CreateActivationFuncion(activation_function_info):
    activation_function, params = Unpack(activation_function_info)
    match activation_function:
        case ActivationFunction.ReLU:
            return lambda x: F.relu(x, *params)
        case ActivationFunction.LeakyReLU:
            return lambda x: F.leaky_relu(x, negative_slope=0.01, *params)
        case ActivationFunction.SoftMax:
            return lambda x: F.softmax(x, dim=1, *params)
        case ActivationFunction.sigmoid:
            return lambda x: F.sigmoid(x, *params)
        
def CreateParamInitializer(initializer):
    def ConvertInitialise(initialise_info):
        initialise, params = Unpack(initialise_info)
        match initialise:
            case ParamInitializer.kaiming_uniform:
                return lambda tensor : nn.init.kaiming_uniform_(tensor, *params)

    # don't do any initialisations when no initialiser is given
    if initializer is None:
        return None

    # if a list of initialisers is given, assume every layer in the cnn has a separate initialiser
    if isinstance(initializer, list):
        new_initializer = []
        for initialise in initializer:
            new_initializer.append(ConvertInitialise(initialise))
        return new_initializer
    
    # if 1 initialiser is given, use it for every layer
    return ConvertInitialise(initializer)

def CreateLossFunction(loss_function_info):
    loss_function, params = Unpack(loss_function_info)
    match loss_function:
        case LossFunction.cross_entropy:
            return nn.CrossEntropyLoss(*params)

def CreateOptimizer(cnn, optimizer_info):
    optimizer, params = Unpack(optimizer_info)
    match optimizer:
        case Optimizer.adam:
            return optim.Adam(cnn.parameters(), *params)

# info is expected to be of type '(enum, [p1, p2, .... pn])' where pi is the ith parameter
# this function converts the following to the above type:
    # info = 'enum' 
        # this becomes '(enum, [])'
    # info = '(enum, p)' where p is a singular parameter
        # this becomes '(enum, [p])'
    # info = '(enum, p1, p2, ..., pn)'
        # this becomes '(enum, [p1, p2, ..., pn])'
def Unpack(info):
    if isinstance(info, tuple):
        # unpacks '(enum, p)' where p can be anything, including a list
        if len(info) == 2:
            enum, params = info
        # turns '(enum, p1, p2, ... pn)' into the expected format
        else:
            enum = info[0]
            params = []
            for i in range(len(info) - 1):
                params.append(info[i+1])
    # turns 'enum' into '(enum, [])'
    else:
        enum = info
        params = []

    # turns '(enum, p)' into '(enum, [p1])'
    if not isinstance(params, list):
        params = [params]

    return (enum, params)