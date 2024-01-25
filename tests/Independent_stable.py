import math
import torch
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood

class MTGP:
    def __init__(self, num_tasks, lr=0.1, training_iterations=30):
        self.num_tasks = num_tasks
        self.lr = lr
        self.training_iterations = training_iterations
        self.models_initialized = False

    def _initialize_models(self, train_x):
        self.models = []
        self.likelihoods = []
        for i in range(self.num_tasks):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x, train_x[:, i], likelihood)
            self.models.append(model)
            self.likelihoods.append(likelihood)

        self.model_list = gpytorch.models.IndependentModelList(*self.models)
        self.likelihood_list = gpytorch.likelihoods.LikelihoodList(*self.likelihoods)
        self.mll = SumMarginalLogLikelihood(self.likelihood_list, self.model_list)
        self.models_initialized = True

    def train(self, train_x):
        if not self.models_initialized:
            self._initialize_models(train_x)

        self.model_list.train()
        self.likelihood_list.train()
        optimizer = torch.optim.Adam(self.model_list.parameters(), lr=self.lr)

        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self.model_list(*[train_x for _ in range(self.num_tasks)])
            targets = [train_x[:, j] for j in range(self.num_tasks)]
            loss = -self.mll(output, targets)
            loss.backward()
            optimizer.step()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, self.training_iterations, loss.item()))

    def get_graph(self, x):
        for model in self.model_list.models:
            model.eval()

        J = torch.zeros(len(x), self.num_tasks, 10)
        for i in range(len(x)):
            print(f"Computing Jacobian for data point {i}")
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

num_tasks = 10
combined_train_x = torch.randn(1000, 10)

mtgp = MTGP(num_tasks)
mtgp.train(combined_train_x)
rms_jacobian, J = mtgp.get_graph(combined_train_x)
print("Adj. Matrix:", rms_jacobian.shape)

# import math
# import torch
# import gpytorch
# from matplotlib import pyplot as plt

# from gpytorch.mlls import SumMarginalLogLikelihood

# training_iterations = 30

# combined_train_x = torch.randn(1000, 10)
# # combined_train_y = torch.randn(1000, 10)

# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# models, likelihoods = [], []
# for i in range(10):
#     likelihood = gpytorch.likelihoods.GaussianLikelihood()
#     model = ExactGPModel(combined_train_x, combined_train_x[:, i], likelihood)
#     models.append(model)
#     likelihoods.append(likelihood)

# model_list = gpytorch.models.IndependentModelList(*models)
# likelihood_list = gpytorch.likelihoods.LikelihoodList(*likelihoods)

# mll = SumMarginalLogLikelihood(likelihood_list, model_list)
# model_list.train()
# likelihood_list.train()

# optimizer = torch.optim.Adam(model_list.parameters(), lr=0.1)

# print("Initial Lengthscales:")
# for i, model in enumerate(model_list.models):
#     lengthscale = model.covar_module.base_kernel.lengthscale
#     print(f"Model {i} initial lengthscale: {lengthscale}")

# for i in range(training_iterations):
#     optimizer.zero_grad()
#     output = model_list(*[combined_train_x for _ in range(10)])
#     targets = [combined_train_x[:, j] for j in range(10)]
#     loss = -mll(output, targets)
#     loss.backward()
#     print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
#     optimizer.step()


# print("\nFinal Lengthscales:")
# for i, model in enumerate(model_list.models):
#     lengthscale = model.covar_module.base_kernel.lengthscale
#     print(f"Model {i} final lengthscale: {lengthscale}")


# for model in model_list.models:
#     model.eval()


# J = torch.zeros(1000, 10, 10)


# for i in range(1000):
#     print(f"Computing Jacobian for data point {i}")

#     input_vector = combined_train_x[i].unsqueeze(0).detach().requires_grad_(True)


#     for j, model in enumerate(model_list.models):

#         model.zero_grad()
#         observed_pred = likelihoods[j](model(input_vector))
#         mean = observed_pred.mean
#         mean.backward()
#         J[i, j] = input_vector.grad.clone()

#         input_vector.grad.zero_()


# mean_squared_jacobians = torch.mean(J ** 2, dim=0)
# rms_jacobian = torch.sqrt(mean_squared_jacobians)

# print("RMS Jacobian (10x10 matrix):")
# print(rms_jacobian.shape)