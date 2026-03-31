import torch
import torch.nn as nn

from settings import t_ratio, batch_size, total_prints
from NetworkDefinition import *

__all__ = ['TrainingProfile', 'get_data']

# gets the first instances out of a dataset with the corresponding label
def get_data(data, label, count):
    data.targets = torch.tensor(data.targets)
    data.data, data.targets = data.data[data.targets==label][0:count], data.targets[data.targets==label][0:count]

# blueprint class for a training profile when training data
class TrainingProfile():
    def __init__(self, cnn : nn.Module, loss_function, optimizer):
        self.loss_function = CreateLossFunction(loss_function)
        self.optimizer = CreateOptimizer(cnn, optimizer)
        self.cnn = cnn

    # the basic training steps, should only be called within this class
    def CoreTraining(self, training_loader, print_func = None):
        # code inspired by: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for i, data in enumerate(training_loader):
            inputs, labels = data

            self.optimizer.zero_grad()
            outputs = self.cnn(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if print_func is not None:
                print_func(i, loss.item())
    

    # does training on the model, accuracy_func should be None if this function is called outside of this class
    def Train(self, training_data, epochs, accuracy_func = None):
        training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
        print_moment = int(len(training_loader) / total_prints * epochs)
        accuracies = []
        total_current_loss = 0
        def print_func(i, loss):
            nonlocal total_current_loss
            total_current_loss += loss
            if i % print_moment == print_moment - 1:   
                print(f'[{epoch + 1}, {i + 1:5d}] avg batch loss: { total_current_loss  / print_moment:.3f}')
                total_current_loss = 0

        for epoch in range(epochs):
            self.CoreTraining(training_loader, print_func)
            total_current_loss = 0
            if accuracy_func is not None:
                accuracies.append(accuracy_func())
        
        return accuracies

    # evaluates the model with the given training data and validation data
    # outputs accuracies
    # accuracies is a 1d array for the accuracies per epoch
    def Validation(self, training_data, validation_data, epochs):
        def calculate_accuracy():
            validation_labels, validation_prediction, _ = self.Inference(validation_data)
            training_labels, training_prediction, _ = self.Inference(training_data)
            validation_acc = (validation_prediction == validation_labels).float().mean().item()
            training_acc = (training_prediction == training_labels).float().mean().item()
            return training_acc, validation_acc

        accuracies = [calculate_accuracy()]
        accs = self.Train(training_data, epochs, calculate_accuracy)     
        accuracies += accs
        return accuracies   

    # evaluates the model using crossvalidation
    # outputs a 2d arracy of accuracies
    # the first dimension is the fold they are a part of.
    # the second dimension is the epoch within that fold they are a part of
    def CrossValidation(self, training_data, epochs, folds):
        block_length = int(len(training_data) / folds)
        length_vector = []
        total_length = 0
        for i in range(folds - 1):
            length_vector.append(block_length)
            total_length += block_length

        length_vector.append(len(training_data) - total_length)
        blocks = torch.utils.data.random_split(training_data, length_vector)

        total_accuracies = []
        for round in range(folds):
            training_data = []
            validation_data = blocks[round]
            for i in range(folds):
                if i != round:
                    training_data += blocks[i]

            accuracies = self.Validation(training_data, validation_data, epochs)
            total_accuracies.append(accuracies)
            self.cnn.Reset()
            print('Finished fold ', round + 1)

        return total_accuracies
    
    # checks the accuracy of the model with the given test set
    def CheckAccuracy(self, testing_data):
        testing_labels, testing_prediction, certainty = self.Inference(testing_data)
        return (testing_prediction == testing_labels).float().mean().item()

    # input data into the cnn and returns what comes out of it
    def Inference(self, data):
        loader = torch.utils.data.DataLoader(data, batch_size=len(data))
        input, labels = next(iter(loader))

        testing_output = self.cnn(input)
        certainty, testing_prediction = torch.max(testing_output.data, 1)
        return labels, testing_prediction, certainty
    
    # does inference with the data until the given layer (inclusive)
    def InferenceUntil(self, data, layer):
        loader = torch.utils.data.DataLoader(data, batch_size=len(data))
        input, _ = next(iter(loader))
        output = self.cnn.ManualForward(input, layer)
        return output
    
    # trains the model until the accuracy stops improving
    # max_lower_acc: how many subsequent epochs the accuracy of the validation set can lower for before assuming convergence is reached
    def TrainUntilConvergence(self, training_data, validation_data, max_lower_acc):
        validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=len(validation_data))
        validation_input, validation_labels = next(iter(validation_loader))

        training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)

        def calculate_accuracy():
            validation_output = self.cnn(validation_input)
            _, validation_prediction = torch.max(validation_output.data, 1)
            return (validation_prediction == validation_labels).float().mean().item()

        epoch = 0
        epochs_stopped_improving = 0
        prev_accuracy = -1
        cur_accuracy = -1
        total_accs = [calculate_accuracy()]
        while epochs_stopped_improving < max_lower_acc:
            prev_accuracy = cur_accuracy
            self.CoreTraining(training_loader, None)
            cur_accuracy = calculate_accuracy()
            total_accs.append(cur_accuracy)
            epoch += 1

            print('epoch = ', epoch, ', accuracy = ', cur_accuracy)
            if cur_accuracy < prev_accuracy:
                epochs_stopped_improving += 1
            else:
                epochs_stopped_improving = 0

        return total_accs



