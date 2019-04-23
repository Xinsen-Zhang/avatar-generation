


import torch
from config import img_size, img_shape, batch_size, lr, n_epochs, latent_dim, n_critic,channels, n_epochs, clip_value, sample_interval
from utils import next_batch, batch_num
from WGAN import model
from torch.autograd import   Variable
import numpy as np
from torchvision.utils import  save_image
from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from torch import nn
import os
import torchvision



os.makedirs("wgan_faces", exist_ok=True)


img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False




# Initialize generator and discriminator
generator = model.Generator()
discriminator = model.Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.00005)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.00005)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize([0.5], [0.5])])
dataset = torchvision.datasets.ImageFolder('./data/faces',transform = transform)
dataloader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size=batch_size,
    shuffle=True,
)


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(n_epochs):
    # batch = next_batch()
    # for i, (imgs, _) in enumerate(dataloader):
    for i, (imgs,_) in tqdm(enumerate(dataloader)):

        # Configure input
        # real_imgs = Variable(imgs.type(Tensor))
        real_imgs = Variable(imgs.type(Tensor))

        if cuda:
            real_imgs = real_imgs.cuda()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # Train the generator every n_critic iterations
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % sample_interval == 0:
            save_image(gen_imgs.data[:64], "wgan_faces/%d.png" % batches_done, nrow=8, normalize=True)
        batches_done += 1
    if epoch % 10 == 0:
        os.system('git add .')
        os.system("git commit -m '{}轮迭代结束'".format(epoch))
        os.system('git push')
