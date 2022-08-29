import os
import tqdm
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image, make_grid


def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    images = []

    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                #                                                          grad.abs().max()))

        return images


def sampling(args, scorenet, niter):
    scorenet.eval()

    sigmas = np.exp(np.linspace(np.log(1), np.log(0.01), 10))
    grid_size = 5

    imgs = []

    samples = torch.rand(grid_size ** 2, 3, 32, 32).cuda(args.gpu)
    all_samples = anneal_Langevin_dynamics(samples, scorenet, sigmas, 100, 0.00002)

    for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
        sample = sample.view(grid_size ** 2, 3, 32, 32)

        if args.logit_transform:
            sample = torch.sigmoid(sample)

        image_grid = make_grid(sample, nrow=grid_size)
        if i % 10 == 0:
            im = Image.fromarray(
                image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
            imgs.append(im)

        save_image(image_grid, os.path.join('outputs', '{}/image_{}.png'.format(niter, i)), nrow=10)
        # torch.save(sample, os.path.join('outputs', 'image_raw_{}_{}.pth'.format(niter, i)))

    imgs[0].save(os.path.join('outputs', "{}/movie.gif".format(niter)), save_all=True, append_images=imgs[1:], duration=1, loop=0)





