import matplotlib.pyplot as plt
import torch
import os
from tempfile import TemporaryFile
import numpy as np

def plot_accuracies(train_losses, train_accuracies, test_losses, test_accuracies, epochs, str_save=None, save_dir=None, FLAGS=[]):
    print("train Losses")
    print(train_losses)
    print('Train accuracies')
    print(train_accuracies)
    print("Test Loses")
    print(test_losses)
    print("test accuracies")
    print(test_accuracies)

    labels = ["train set", "test set"]
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epochs, train_accuracies, epochs, test_accuracies)
    ax1.set_title('Accuracies')
    ax1.legend(labels)

    ax2.plot(epochs, train_losses, epochs, test_losses)
    ax2.legend(labels)
    ax2.set_title('Loss')


    if str_save != None:
        save_dir += 'Figures/'
        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
        plt.savefig(save_dir + str_save + '.pdf')

        str_save = save_dir + str_save + '.txt'
        write_params(str_save, FLAGS)
    plt.show()
    print('Did plot')
    return

def accuracy(prediction, targets):

    values , predictions = torch.max(prediction, 1)


    ls_corrects = predictions == targets
    hits = torch.sum(ls_corrects.int())

    accuracy_val = (hits.item() / len(ls_corrects))

    return accuracy_val

def save_results(train_losses, train_accuracies, test_losses, test_accuracies, str_save=None, save_dir=None, FLAGS = None):

    target_dir = save_dir + 'TrainingResults/' + str_save + '/'
    if not(os.path.isdir(target_dir)):
        os.mkdir(target_dir)

    np.save(target_dir + 'train_losses', train_losses)
    np.save(target_dir + 'train_accuracies', train_accuracies)
    np.save(target_dir + 'test_losses', test_losses)
    np.save(target_dir + 'test_accuracies', test_accuracies)
    str_params = target_dir + 'params.txt'
    write_params(str_params, FLAGS)

def write_params(str_save, FLAGS):
    F = open(str_save, 'w')
    for key, value in vars(FLAGS).items():
        F.write(key + ' : ' + str(value) + '\n')
    F.close()