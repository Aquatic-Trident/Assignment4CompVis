import torch
import torch.nn as nn


from settings import model_location_information
from NetworkDefinition import *

# this class contains the definition of a CNN
__all__ = ['CNN']


# this class constructs the CNN
class CNN(nn.Module):
    # layers: list of tuples, first elem is a networkLayer enum and the second a list of parameters
    # initialiser: can be None (no initialisation), a function (initialise every layer with the same function) or a list of functions
    # then len(initializer) == len(layers) and initializer contains an initialisation function for every layer
    # layer_activation_func: activation function for every layer in ActivationLayers, is of type ActivationFunction enum
    # output_activation_func: activation function for the output, is of type ActivationFunction enum
    # name: name of the model, used when saving and loading model weights
    def __init__(self, layers, layer_activation_func, output_activation_func, initializer, name):
        super().__init__()

        self.layer_activation_func = CreateActivationFuncion(layer_activation_func)
        self.output_activation_func = CreateActivationFuncion(output_activation_func)
        self.initializer = initializer
        self.name = name

        self.cnn = []
        self.activatable_layers = [] # keep track of which layers need activation functions
        for i in range(len(layers)):
            layer = layers[i]
            layer_func = CreateNetworkLayer(layer)

            # yes this function is necessary. For some reason self.parameters() only updates
            # when you assign layer_func to a self variable (the name doesn't matter, as long as it's self.{var_name})
            # I assume this is because there is a weird python class function that gets called every time you assign
            # a value to a self variable. Through this function, the parent class updates self.parameters()
            # but since I don't care about assigning each to individual variables and I want to store the layer_func
            # values in a list instead, I have to do this. Appending the value to a list or reassigning all layer_func values
            # to the same variable does not work: it only updates self.parameters() once.
            exec('self.l' + str(i) + '= layer_func')          

            self.cnn.append(layer_func)

            layer_, _ = Unpack(layer)
            if layer_ in ActivationLayers:
                self.activatable_layers.append(True)
            else:
                self.activatable_layers.append(False)          
        
        self.Reset()

    # forward function, is part of nn.Module
    def forward(self, x):
        for i in range(len(self.cnn) - 1):
            x = self.cnn[i](x)

            if self.activatable_layers[i]:
                x = self.layer_activation_func(x)
        
        x = self.cnn[-1](x)
        x = self.output_activation_func(x)
        return x 
    
    def ManualForward(self, x, layer):
        for i in range(layer + 1):
            x = self.cnn[i](x)

            if self.activatable_layers[i]:
                x = self.layer_activation_func(x)
                
        return x 
    
    # resets model weights to their initial value according to the value of self.initializer
    def Reset(self):
        initializer = CreateParamInitializer(self.initializer)
        isList = isinstance(initializer, list)
        if initializer is not None:
            for i in range(len(self.cnn)):
                if hasattr(self.cnn[i], 'reset_parameters'):
                    if isList:
                        initializer[i](self.cnn[i].weight)
                    else:
                        initializer(self.cnn[i].weight) 
        else:
            for layer in self.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    # freezes the given layer or layers, such that their parameters don't get updated
    # layer can be a single int or a list of ints. Layer ints correspond to the index of the layer the constructor
    # of this class was given.
    def FreezeLayers(self, layers):
        if isinstance(layers, int):
            layers = [layers]

        for layer in layers:
            if hasattr(self.cnn[layer], '_parameters'):
                if 'weight' in self.cnn[layer]._parameters:
                    self.cnn[layer]._parameters['weight'].requires_grad = False
                if 'bias' in self.cnn[layer]._parameters:
                    self.cnn[layer]._parameters['bias'].requires_grad = False

    # removes the layers with the given index/indices from the model
    # layer can be a single int or a list of ints. Layer ints correspond to the index of the layer the constructor
    # of this class was given.
    def RemoveLayers(self, layers):
        if isinstance(layers, int):
            layers = [layers]

        for layer in layers:
            self.cnn[layer] = nn.Identity()
            exec('self.l' + str(layer) + ' = nn.Identity()')

    # replaces the layers with the given index/indices in the model with the given layers
    # layer can be a single int or a list of ints. Layer ints correspond to the index of the layer the constructor
    # of this class was given.
    def ReplaceLayers(self, layers, replacements):
        if isinstance(layers, int):
            layers = [layers]
            replacements = [replacements]            

        _replacements = []
        for replacement in replacements:
            _replacements.append(CreateNetworkLayer(replacement))

        for i in range(len(layers)):
            self.cnn[layers[i]] = _replacements[i]
            exec('self.l' + str(layers[i]) + ' = _replacements[i]')

    def GetLayer(self, layer):
        return self.cnn[layer].weight.data

    # saves the model to a file
    def SaveWeights(self, index):
        model_save_folder, model_save_extension = model_location_information
        path = model_save_folder + self.name + '-' + str(index) + model_save_extension
        torch.save(self.state_dict(), path)

    # loads the model from a file
    def LoadWeights(self, index):
        model_save_folder, model_save_extension = model_location_information
        path = model_save_folder + self.name + '-' + str(index) + model_save_extension
        self.load_state_dict(torch.load(path, weights_only=False))
        self.eval()
        
    # loads the model from a file by filename
    def LoadWeightsByName(self, filename):
        model_save_folder, model_save_extension = model_location_information
        path = model_save_folder + filename + model_save_extension
        self.load_state_dict(torch.load(path, weights_only=False))
        self.eval()



    


