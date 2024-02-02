import math
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.mlls import SumMarginalLogLikelihood

training_iterations = 100
combined_train_x = torch.randn(1000, 10)
combined_train_y = torch.randn(1000, 10)   


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


models, likelihoods = [], []
for i in range(10):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(combined_train_x[i], combined_train_y[i], likelihood)
    models.append(model)
    likelihoods.append(likelihood)

model_list = gpytorch.models.IndependentModelList(*models)
likelihood_list = gpytorch.likelihoods.LikelihoodList(*likelihoods)

mll = SumMarginalLogLikelihood(likelihood_list, model_list)
model_list.train()
likelihood_list.train()

# Use the Adam optimizer
optimizer = torch.optim.Adam(model_list.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

print("Initial Lengthscales:")
for i, model in enumerate(model_list.models):
    lengthscale = model.covar_module.base_kernel.lengthscale
    print(f"Model {i} initial lengthscale: {lengthscale}")

losses = []
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model_list(*model_list.train_inputs)
    loss = -mll(output, model_list.train_targets)
    loss.backward()
    losses.append(loss.item())
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()

print("\nFinal Lengthscales:")
for i, model in enumerate(model_list.models):
    lengthscale = model.covar_module.base_kernel.lengthscale
    print(f"Model {i} final lengthscale: {lengthscale}")


##################################################


class Dagma_DCE_Module(nn.Module, abc.ABC):
    @abc.abstractmethod
    def get_graph(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def h_func(self, W: torch.Tensor, s: float) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_l1_reg(self, W: torch.Tensor) -> torch.Tensor:
        ...


class DagmaGP_DCE(Dagma_DCE_Module):

    def __init__(self, X: torch.Tensor):

        super(DagmaGP_DCE, self).__init__()
        self.X = X
        self.n, self.d = X.shape
        self.I = torch.eye(self.d)

        models, likelihoods = [], []
        for i in range(self.d):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(X, X[:i], likelihood)
            models.append(model)
            likelihoods.append(likelihood)

        model_list = gpytorch.models.IndependentModelList(*models)
        likelihood_list = gpytorch.likelihoods.LikelihoodList(*likelihoods)

        mll = SumMarginalLogLikelihood(likelihood_list, model_list)
        model_list.train()
        likelihood_list.train()



    def h_func(self, W: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        """Calculate the DAGMA constraint function

        Args:
            W (torch.Tensor): adjacency matrix
            s (float, optional): hyperparameter for the DAGMA constraint,
                can be any positive number. Defaults to 1.0.

        Returns:
            torch.Tensor: constraint
        """
        h = -torch.slogdet(s * self.I - W * W)[1] + self.d * np.log(s)
        return h

    def get_l1_reg(self, observed_derivs: torch.Tensor) -> torch.Tensor:
        """Gets the L1 regularization

        Args:
            observed_derivs (torch.Tensor): the batched Jacobian matrix

        Returns:
            torch.Tensor: _description_
        """
        return torch.sum(torch.abs(torch.mean(observed_derivs, axis=0)))
    

# fig, ax = plt.subplots(figsize=(10, 5))

# ax.plot(losses)
# ax.set(title="Loss", xlabel="Iterations", ylabel="Negative Log-Likelihood")
# plt.show()

# model_list.eval()
# likelihood_list.eval()

# # Initialize plots
# f, axs = plt.subplots(1, 3, figsize=(8, 3))

# # Make predictions (use the same test points)
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     test_x = torch.linspace(0, 1, 50)
#     # This contains predictions for both outcomes as a list
#     predictions = likelihood_list(*model_list(test_x, test_x, test_x))

# for submodel, prediction, ax in zip(model_list.models, predictions, axs):
#     mean = prediction.mean
#     var = prediction.variance
#     std = torch.sqrt(var.squeeze())
#     lower, upper = prediction.confidence_region()

#     tr_x = submodel.train_inputs[0].detach().numpy()
#     tr_y = submodel.train_targets.detach().numpy()

#     # Plot training data as black stars
#     ax.plot(tr_x, tr_y, 'k*')
#     # Predictive mean as blue line
#     ax.plot(test_x.numpy(), mean.numpy(), 'b')
#     # Shade in confidence
#     ax.fill_between(test_x.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
#     ax.set_ylim([-3, 3])
#     ax.legend(['Observed Data', 'Mean', 'Confidence'])
#     ax.set_title('Observed Values (Likelihood)')

# plt.show()

# X = torch.autograd.Variable(torch.Tensor(test_x), requires_grad=True)
# observed_pred = likelihood(model(X))
# dydtest_x_ag  = torch.autograd.grad(observed_pred.mean.sum(), X)[0]

# fig, ax = plt.subplots(figsize=(16,5))
# # ax.plot(xtest_tensor.detach().numpy(), mu, color="black", label="Mean")
# # ax.plot(xtest_tensor.detach().numpy(), dydtest_x.detach().numpy(), 'green', label="Gradient (Tensor)")
# ax.plot(test_x.detach().numpy(), dydtest_x_ag.detach().numpy(), 'orange', linestyle="-.", label="Gradient (Autograd)")
# # ax.plot(xtest_tensor.detach().numpy(), dydtest_x_f.detach().numpy(), 'red', linestyle="--", label="Gradient (Functional)")
# plt.legend()
# plt.show()

# print(f"size of dydtest_x_ag: {dydtest_x_ag.size()}")
# print(f"dydtest_x_ag: {dydtest_x_ag}")