import numpy as np 
import torch
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
def calc_accuracy(out, targets, one_hot=(True, False)):
    #print('Out size: {0}; target size: {1}'.format(out.size(), targets.size()))
    batch_size = out.size()[0]
    if one_hot[0]: 
        #print(out)
        _, out = out.max(1)
        #print(out)
    if one_hot[1]: 
        _, targets = targets.max(1)
    
    #print('Out size: {0}; target size: {1}'.format(out.size(), targets.size()))
    hits = (out == targets).to(torch.float) * 1
    #print(hits)
    accuracy = hits.sum() / batch_size

    return accuracy

def save_results(train_losses, train_accuracies, test_loss, epochs, str_save=None, save_dir=None, FLAGS = None):

    target_dir = save_dir + '/TrainingResults/' + str_save + '/'
    if not(os.path.isdir(target_dir)):
        os.mkdir(target_dir)

    np.save(target_dir + 'train_losses', train_losses)
    np.save(target_dir + 'train_accuracies', train_accuracies)
    if test_loss is not None:
        np.save(target_dir + 'test_losses', test_loss)
    np.save(target_dir + 'epochs', epochs)

def plot_accuracies(train_losses, train_accuracies, test_accuracy, epochs, str_save=None, save_dir=None, FLAGS=[]):

    f, ax1 = plt.subplots(1, 1)

    color = 'tab:blue'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(epochs, train_accuracies, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    if test_accuracy is not None:
        ax1.plot(epochs[-1], test_accuracy, color='tab:green', marker='D')

    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax3.set_ylabel('Loss', color=color)  
    ax3.plot(epochs, train_losses, color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_title('Train accuracies and losses')



    if str_save != None:
        save_dir += 'Figures/'
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        plt.savefig(save_dir + str_save + '.pdf')

        str_save = save_dir + str_save + '.txt'
        
    plt.show()
    print('Did plot')
    return
def bar_graphs(input_lengths, accuracies, title_string):
    print(input_lengths)
    print(accuracies)
    x = np.arange(len(input_lengths))+1
    f = matplotlib.pyplot.bar(x, accuracies, tick_label=input_lengths )
    plt.title(title_string)
    plt.show()
    
if __name__ == '__main__':
    input_length = [5, 10, 21, 40, 81, 160, 321]
    rnn_accuracies = [1, 1, 1, 0.32, 0.2, 0.2, 0.18]

    bar_graphs(input_length, rnn_accuracies, 'RNN accuracies')

    input_length = [5, 10, 21, 40, 81, 160, 321, 640]
    lstm_accuracies = [1, 1, 1, 1, 1, 1, 1, 0.1]

    bar_graphs(input_length, lstm_accuracies, 'LSTM accuracies')
