
import copy
import torch
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import copy
from tqdm.auto import tqdm
import abc
import typing
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood

class MTGPModel:
    def __init__(self, input_dim, output_dim, num_tasks, lr=0.1, training_iterations=30):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tasks = num_tasks
        self.lr = lr
        self.training_iterations = training_iterations

        # Initialize models and likelihoods for each task
        self.models = []
        self.likelihoods = []
        for _ in range(num_tasks):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(input_dim, output_dim, likelihood)
            self.models.append(model)
            self.likelihoods.append(likelihood)

        # Combine models into a model list
        self.model_list = gpytorch.models.IndependentModelList(*self.models)
        self.likelihood_list = gpytorch.likelihoods.LikelihoodList(*self.likelihoods)
        self.mll = SumMarginalLogLikelihood(self.likelihood_list, self.model_list)
    
    def train(self, train_x, train_y):
        # Set models to training mode
        self.model_list.train()
        self.likelihood_list.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.model_list.parameters(), lr=self.lr)

        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self.model_list(*[train_x for _ in range(self.num_tasks)])
            targets = [train_y[:, j] for j in range(self.num_tasks)]
            loss = -self.mll(output, targets)
            loss.backward()
            optimizer.step()

            print(f'Iter {i+1}/{self.training_iterations} - Loss: {loss.item()}')

    def get_graph(self, x):
        # Set models to evaluation mode
        for model in self.model_list.models:
            model.eval()

        J = torch.zeros(len(x), self.num_tasks, self.input_dim)

        for i in range(len(x)):
            input_vector = x[i].unsqueeze(0).detach().requires_grad_(True)

            for j, model in enumerate(self.model_list.models):
                model.zero_grad()
                observed_pred = self.likelihoods[j](model(input_vector))
                mean = observed_pred.mean
                mean.backward()
                J[i, j] = input_vector.grad.clone()
                input_vector.grad.zero_()

        mean_squared_jacobians = torch.mean(J ** 2, dim=0)
        rms_jacobian = torch.sqrt(mean_squared_jacobians)

        return rms_jacobian, J


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Example usage:
mtgp = MTGPModel(input_dim=10, output_dim=1, num_tasks=10)
combined_train_x = torch.randn(1000, 10)
combined_train_y = torch.randn(1000, 10)
mtgp.train(combined_train_x, combined_train_y)
rms_jacobian, J = mtgp.get_graph(combined_train_x)
print("RMS Jacobian (10x10 matrix):", rms_jacobian.shape)