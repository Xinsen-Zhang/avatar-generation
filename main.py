
# 数据的检查
import os
if os.path.exists('./data'):
    print('data prepared')
else:
    _ = os.popen('gunzip -c data.tar.gz > data.tar')
    _ = os.popen('tar -xvf data.tar')
    _ = os.popen('rm -f ./data.tar')
    print('data prepared')

# import packages
import torch
from config import img_size, img_shape, batch_size, lr, n_epochs, latent_dim, n_critic,channels, n_epochs, clip_value, sample_interval
# from utils import next_batch, batch_num
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

# preprocess
img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False


# 模型的加载

def load_model():
    generator = torch.load('./checkpoint/generator.pkl')
    discriminator = torch.load('./checkpoint/discriminator.pkl')
    optimizer_D = torch.load('./checkpoint/optimizer_D.pkl')
    optimizer_G = torch.load('./checkpoint/optimizer_G.pkl')
    return generator, discriminator, optimizer_G, optimizer_D

if os.path.exists('./checkpoint'):
    pass
    generator, discriminator, optimizer_G, optimizer_D = load_model()
else:
    # Initialize generator and discriminator
    os.mkdir('./checkpoint')
    generator = model.Generator()
    discriminator = model.Discriminator()

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.00005)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.00005)
if cuda:
    generator.cuda()
    discriminator.cuda()

# dataloader
transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = torchvision.datasets.ImageFolder('./data/faces', transform = transform)
dataloader = torch.utils.data.DataLoader(
    dataset= dataset,
    shuffle= True,
    batch_size= batch_size
)

# define tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(600, 600 + n_epochs):
    # load the model
    if epoch % 100 == 0:
        if os.path.exists('./checkpoint'):
            try:
                generator, discriminator, optimizer_G, optimizer_D = load_model()
            except Exception as e:
                pass
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

            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #     % (epoch, n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
            # )

        # if batches_done % sample_interval == 0:
        batches_done += 1
    save_image(gen_imgs.data[:25], "wgan_faces/epoch_{}.png".format(epoch), nrow=5, normalize=True)
    
    print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, loss_D.item(), loss_G.item())
            )

    if epoch % 20 == 0:
        torch.save(generator, './checkpoint/generator.pkl')
        torch.save(discriminator, './checkpoint/discriminator.pkl')
        torch.save(optimizer_D, './checkpoint/optimizer_D.pkl')
        torch.save(optimizer_G, './checkpoint/optimizer_G.pkl')
        os.system('git add .')
        os.system("git commit -m '{}轮迭代结束'".format(epoch))
        os.system('git push')

