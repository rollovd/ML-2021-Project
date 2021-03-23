import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

import ot
from sklearn.metrics import pairwise_distances


class GPLoss(nn.Module):
    def __init__(self, lambda_gp):
        super(GPLoss, self).__init__()

        self.optimizer_params = {'lr': 0.0002, 'betas': (0, 0.9)}
        self.lambda_gp = lambda_gp

    def forward(self, real_samples, fake_samples, D, phi, psi, device):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.randn(real_samples.size(0), 1, 1, 1).to(device)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        d_interpolates = D(interpolates).view(real_samples.size(0), 1, 1, 1)
        fake = Variable(torch.randn(real_samples.shape[0], 1, 1, 1).fill_(1.0).to(device), requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return - torch.mean(phi) + torch.mean(psi) + self.lambda_gp * gradient_penalty


class WCLoss(nn.Module):

    def __init__(self):
        super(WCLoss, self).__init__()
        self.optimizer_params = {'lr': 5e-5}

    def forward(self, real, fake):
        return torch.mean(real) - torch.mean(fake)


class WeightClipper(object):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.epsilon, self.epsilon)
            module.weight.data = w

class CEpsilonLoss(nn.Module):

    def __init__(self, epsilon=0.1):
        super(CEpsilonLoss, self).__init__()
        self.epsilon = epsilon
        self.optimizer_params = {'lr': 1e-4}

    def ot_sink(self, objects):
        hist = torch.ones(objects.size(0)) / objects.size(0)
        norms = torch.Tensor([torch.norm(x) for x in objects])
        M = pairwise_distances(norms.view(-1, 1), norms.view(-1, 1), metric='l1')
        opt_t = (ot.sinkhorn(hist, hist, M, 1) * M).sum()

        return opt_t

    def forward(self, real_objects, fake_objects, fake_validity):
        real = real_objects.view(real_objects.size(0), -1)
        fake = fake_objects.view(fake_objects.size(0), -1)

        c_matrix = torch.cdist(real, fake, p=1.0)
        fake_term = torch.mean(fake_validity)

        term2 = (fake_validity - c_matrix) * self.epsilon

        term2 = torch.exp(term2).mean(1)
        term2 = torch.log(term2)

        return -fake_term + term2.mean() / self.epsilon


class CLoss(nn.Module):

    def __init__(self):
        super(CLoss, self).__init__()

        self.optimizer_params = {'lr': 0.001}

    def forward(self, real_objects, fake_objects, fake_validity):
        real = real_objects.view(real_objects.size(0), -1)
        fake = fake_objects.view(fake_objects.size(0), -1)

        c_matrix = torch.cdist(real, fake, p=1.0)
        fake_term = torch.mean(fake_validity)

        term2 = torch.min(c_matrix - fake_validity, dim=1)[0]

        return -fake_term - torch.mean(term2)


class LPLoss(nn.Module):
    def __init__(self, lambda_gp):
        super(LPLoss, self).__init__()
        self.optimizer_params = {'lr': 0.0002, 'betas': (0, 0.9)}
        self.lambda_gp = lambda_gp

    def forward(self, real_samples, fake_samples, D, phi, psi, device):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.randn(real_samples.size(0), 1, 1, 1).to(device)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates).view(real_samples.size(0), 1, 1, 1)
        fake = Variable(torch.randn(real_samples.shape[0], 1, 1, 1).fill_(1.0).to(device), requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        lipschitz_penalty = (torch.maximum(torch.Tensor([0]).to(device), gradients.norm(2, dim=1) - 1) ** 2).mean()

        loss = -torch.mean(phi) + torch.mean(psi) + self.lambda_gp * lipschitz_penalty
        return loss
