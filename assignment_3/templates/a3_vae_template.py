import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        #MNIST input size is 28x28
        self.fc = nn.Linear(28*28, hidden_dim)
        self.means = nn.Linear(hidden_dim, z_dim)
        self.stds = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        #with relu nonlinearity
        hidden_state = self.fc(input).tanh()

        mean = self.means(hidden_state)
        std = self.stds(hidden_state)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        #mirrors the encoder
        self.fc = nn.Linear(z_dim, hidden_dim)
        #again output size is MNIST 
        self.out = nn.Linear(hidden_dim, 28*28)

    def forward(self, input_):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        hidden = self.fc(input_).relu()

        mean = self.out(hidden).sigmoid()

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.loss_bce = nn.BCELoss(reduction='none')

    def forward(self, input_):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        input_ = input_.view(-1, 28*28)
        batch_size = input_.size()[0]

        #get latent variables
        z_mean, z_std = self.encoder(input_)

        #create a sample from latent space 
        sample = torch.randn(*z_mean.size())
        sample = sample.to(device)

        #we wanna do the reparameterization trick here 
        z_std = z_std.exp().sqrt()
        sample = sample * z_std + z_mean

        #run the decoding step to get mean for bernoulli of out
        out = self.decoder(sample)

        #now we can calculate the los terms

        l_recon = self.loss_bce(out, input_).sum()
       
        l_kl = 0.5 * (z_mean**2  + (z_std ** 2).log() -1 ).sum()

        #add up with averaging over batches
        average_negative_elbo = (l_recon + l_kl) / batch_size
        
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        
        #get random samples 
        sample = torch.randn((n_samples, self.z_dim))
        sample = sample.to(device)

        #we wanna do the reparameterization trick here 
        im_means = self.decoder(sample)
        
        sampled_ims = torch.bernoulli(im_means).view(-1, 28,28)
        
        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    
    for enum, input_ in enumerate(data):
        
        input_ = input_.view(-1, 28*28)
        
        input_ = input_.to(device)

        optimizer.zero_grad()

        average_epoch_elbo = model(input_)

        if model.training:
            average_epoch_elbo.backward()
            optimizer.step()

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train() #? 
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
   
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if (epoch % ARGS.eval_interval) == 0 : 
            n_samples = 4*4
            img_samples, im_means = model.sample(n_samples)
            im_means = im_means.view(-1,28,28)
            img = im_means.detach().cpu().numpy()
            
            for i in range(n_samples):
                plt.subplot(4,4,i+1)
                fig = plt.imshow(img[i,:,:], cmap='Greys')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
            str_save = 'figures/VAE_samples_epoch_' + str(epoch)
            plt.savefig(str_save)
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--device', type=str, default="cuda:0", 
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--eval_interval', type=int, default=10, 
                        help="Training device 'cpu' or 'cuda:0'")


    ARGS = parser.parse_args()

    device = torch.device(ARGS.device)
    
    main()
