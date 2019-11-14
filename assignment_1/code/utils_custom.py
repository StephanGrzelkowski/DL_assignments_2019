import matplotlib.pyplot as plt
import torch

def plot_accuracies(train_losses, train_accuracies, test_losses, test_accuracies, epochs):
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

    plt.show()
    print('Did plot')
    return

def accuracy(prediction, targets):

    values , predictions = torch.max(prediction, 1)
    print("Predictions: ")
    print(predictions)
    print("targets")
    print(targets)
    ls_corrects = predictions == targets
    hits = torch.sum(ls_corrects.int())

    print(hits)
    print(len(ls_corrects))
    accuracy_val = (hits.item() / len(ls_corrects))

    return accuracy_val
