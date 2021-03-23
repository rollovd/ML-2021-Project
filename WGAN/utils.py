import torch
import torch.utils.data
import numpy as np
from tqdm.notebook import tqdm
import ot
from sklearn.metrics import pairwise_distances
import numpy as np

def get_gt_wass(fake_objects, real_objects):
    fake_hist = torch.ones(fake_objects.shape[0]) / fake_objects.shape[0]
    real_hist = torch.ones(fake_objects.shape[0]) / fake_objects.shape[0]

    M = pairwise_distances(fake_objects.flatten(start_dim=1).cpu().detach().numpy(), real_objects.flatten(start_dim=1).cpu().detach().numpy(), metric='l1')
    wass = (ot.emd(fake_hist, real_hist, M) * M).sum()
    
    return wass

def divide_by_classes(dataset, batch_size, cifar=True, stability_check=False):
    part1 = []
    part2 = []

    i = j = 0

    for data, target in dataset:
        if target < 5:
            if not stability_check or stability_check and i < 512:
                part1.append(data)
                i += 1
        else:
            if not stability_check or stability_check and j < 512:
                part2.append(data)
                j += 1

    images_part1 = torch.stack(part1) / 255
    images_part2 = torch.stack(part2) / 255

    if not cifar:
        min_num = min(images_part1.size(0), images_part2.size(0))
        images_part1 = images_part1[:min_num, ...]
        images_part2 = images_part2[:min_num, ...]

    loader_part1 = torch.utils.data.DataLoader(images_part1, batch_size=batch_size,
                                         shuffle=True)

    loader_part2 = torch.utils.data.DataLoader(images_part2, batch_size=batch_size,
                                         shuffle=True)

    return loader_part1, loader_part2


class Trainer:
    def __init__(self, criterion, train_part1, train_part2, test_part1, test_part2, discriminator, weight_clipper=None):
        self.criterion = criterion
        self.train_part1 = train_part1
        self.train_part2 = train_part2
        self.test_part1 = test_part1
        self.test_part2 = test_part2
        self.discriminator = discriminator
        self.weight_clipper = weight_clipper

    def train(self, epochs, device, validate=True, verbose=True):

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        params = self.criterion.optimizer_params
        optimizer = torch.optim.RMSprop(self.discriminator.parameters(), **params) if self.criterion.__class__.__name__ not in ['GPLoss', 'LPLoss'] \
                    else torch.optim.Adam(self.discriminator.parameters(), **params)

        freq = max(epochs // 20, 1)

        error_val = []
        val_dest = []
        val_ground = []

        for epoch in range(epochs):
            self.discriminator.train()
            losses_train = []   

            for i, (imgs1, imgs2) in tqdm(enumerate(zip(self.train_part1, self.train_part2))):

                imgs_m = imgs1.to(device, dtype=torch.float32)
                imgs_v = imgs2.to(device, dtype=torch.float32)
              
                optimizer.zero_grad() 

                phi = self.discriminator(imgs_m).view(-1)
                psi = self.discriminator(imgs_v).view(-1)

                if self.criterion.__class__.__name__ == 'WCLoss':
                    loss = self.criterion(phi, psi) # WCLoss
                elif self.criterion.__class__.__name__ == 'GPLoss' or self.criterion.__class__.__name__ == 'LPLoss':
                    loss = self.criterion(imgs_m, imgs_v, self.discriminator, phi, psi, device) # GPLoss
                elif self.criterion.__class__.__name__ == 'CLoss' or self.criterion.__class__.__name__ == 'CEpsilonLoss':
                    loss = self.criterion(imgs_m, imgs_v, psi) # CEpsilonLoss, CLoss
                
                loss.backward()
                losses_train.append(loss)
                optimizer.step()

                if self.criterion.__class__.__name__ == 'WCLoss':
                    self.discriminator.apply(self.weight_clipper)

            if validate:
                self.discriminator.eval() 
                if verbose and epoch%freq==0:

                    for i, (imgs1, imgs2) in tqdm(enumerate(zip(self.test_part1, self.test_part2))):
                        
                        imgs_m = imgs1.to(device, dtype=torch.float32)
                        imgs_v = imgs2.to(device, dtype=torch.float32)

                        phi = self.discriminator(imgs_m).view(-1)
                        psi = self.discriminator(imgs_v).view(-1)

                        if self.criterion.__class__.__name__ == 'WCLoss':
                            d_est = -self.criterion(phi, psi) # WCLoss
                        elif self.criterion.__class__.__name__ == 'GPLoss' or self.criterion.__class__.__name__ == 'LPLoss':
                            d_est = -self.criterion(imgs_m, imgs_v, self.discriminator, phi, psi, device) # GPLoss
                        elif self.criterion.__class__.__name__ == 'CLoss':
                            d_est = -self.criterion(imgs_m, imgs_v, psi) # CLoss
                        elif self.criterion.__class__.__name__ == 'CEpsilonLoss':
                            d_est = -self.criterion(imgs_m, imgs_v, psi) - (self.criterion.ot_sink(imgs_m) + self.criterion.ot_sink(imgs_v)) / 2 # CEpsilonLoss

                        val_dest.append(d_est.item())

                        d_ground = get_gt_wass(imgs_m, imgs_v)
                        val_ground.append(d_ground)

                        error = d_est - d_ground
                        error_val.append(error.item())

                    mean_error_val = sum(error_val)/len(error_val)

                    print('Val epoch {}'.format(epoch), \
                      ', Mean error : {:.8}'.format(mean_error_val))
                
                torch.cuda.empty_cache()

        return val_dest, val_ground #our estimation, ground