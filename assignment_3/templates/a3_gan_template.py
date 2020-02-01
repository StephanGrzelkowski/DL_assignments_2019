import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        self.fc1 = nn.Linear(args.latent_dim, 128)
        #   LeakyReLU(0.2)
        self.rl1 = nn.LeakyReLU(0.2)
        #   Linear 128 -> 256
        self.fc2 = nn.Linear(128, 256)
        #   Bnorm
        self.bn2 = nn.BatchNorm1d(256)
        #   LeakyReLU(0.2)
        self.rl2 = nn.LeakyReLU(0.2)
        #   Linear 256 -> 512
        self.fc3 = nn.Linear(256, 512)
        #   Bnorm
        self.bn3 = nn.BatchNorm1d(512)
        #   LeakyReLU(0.2)
        self.rl3 = nn.LeakyReLU(0.2)
        #   Linear 512 -> 1024
        self.fc4 = nn.Linear(512, 1024)
        #   Bnorm
        self.bn4= nn.BatchNorm1d(1024)
        #   LeakyReLU(0.2)
        self.rl4 = nn.LeakyReLU(0.2)
        #   Linear 1024 -> 768
        self.fc5 = nn.Linear(1024, 784)
        #   Output non-linearity
        self.sig = nn.Sigmoid()

    def forward(self, z):
        # Generate images from z
        x = self.rl1(self.fc1(z))
        x = self.rl2(self.bn2(self.fc2(x)))
        x = self.rl3(self.bn3(self.fc3(x)))
        x = self.rl4(self.bn4(self.fc4(x)))
        out = self.sig(self.fc5(x))

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        self.fc1 = nn.Linear(784, 512)
        #   LeakyReLU(0.2)
        self.rl1 = nn.LeakyReLU(0.2)

        #   Linear 512 -> 256
        self.fc2 = nn.Linear(512, 256)
        #   LeakyReLU(0.2)
        self.rl2 = nn.LeakyReLU(0.2)

        #   Linear 256 -> 1
        self.fc3 = nn.Linear(256, 1)
        #   Output non-linearity
        self.sig = nn.Sigmoid()

    def forward(self, img):
        # return discriminator score for img
        x = self.rl1(self.fc1(img))
        x = self.rl2(self.fc2(x))
        out = self.sig(self.fc3(x))

        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.view(-1, 28*28)
            imgs = imgs.to(device)

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            #get random input and generate images from it
            rand_input = torch.randn(args.batch_size, args.latent_dim).to(device)
            #run image generator and then discriminator 
            imgs_gen = generator(rand_input)

            pred = discriminator(imgs_gen)
            
            #get the generator loss
            loss_gen = -torch.sum(torch.log(pred)) / args.batch_size #we don't want the generator to care about the real images 

            loss_gen.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            #run the discriminator for real and fake images 
            optimizer_D.zero_grad()
            pred_real = discriminator(imgs)
            rand_input = torch.randn(args.batch_size, args.latent_dim).to(device)
            imgs_gen = generator(rand_input)
            pred_fake = discriminator(imgs_gen)

            #compute losses and run gradient
            loss_dis = (torch.sum(torch.log(pred_real)) + torch.sum(torch.log(1.0 - pred_fake)) ) / (args.batch_size)
            loss_dis.backward()
            optimizer_D.step()


        # Save Images
        # -----------
        if epoch % args.save_interval == 0:
            # You can use the function save_image(Tensor (shape Bx1x28x28),
            # filename, number of rows, normalize) to save the generated
            # images, e.g.:
            # save_image(gen_imgs[:25],
            #            'images/{}.png'.format(batches_done),
            #            nrow=5, normalize=True)
            gen_imgs = imgs_gen.view(-1, 1, 28, 28)
            save_image(gen_imgs[:25],
                'figures/GAN/{}.png'.format(epoch),
                nrow=5, normalize=True)
        print('Done with Epoch: {}'.format(epoch))

def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ),
                                                (0.5, ))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "models/mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Training device "cpu" or "cuda:0"')

    args = parser.parse_args()
    device =  torch.device(args.device)
    main()
