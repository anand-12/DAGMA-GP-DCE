#@title Load Packages
# TYPE HINTS
from typing import Tuple, Optional, Dict, Callable, Union

# PyTorch Settings
import torch

# Pyro Settings

# GPyTorch Settings
import gpytorch

# PyTorch Lightning Settings

# NUMPY SETTINGS
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# MATPLOTLIB Settings
import matplotlib as mpl
import matplotlib.pyplot as plt


# SEABORN SETTINGS
import seaborn as sns
sns.set_context(context='talk',font_scale=0.7)
# sns.set(rc={'figure.figsize': (12, 9.)})
# sns.set_style("whitegrid")

# PANDAS SETTINGS
import pandas as pd
pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

# LOGGING SETTINGS
import tqdm
import wandb

def regression_near_square(
    n_train: int = 50,
    n_test: int = 1_000,
    x_noise: float = 0.3,
    y_noise: float = 0.2,
    seed: int = 123,
    buffer: float = 0.1,
):
    rng = np.random.RandomState(seed)

    # function
    f = lambda x: np.sin(1.0 * np.pi / 1.6 * np.cos(5 + 0.5 * x))

    # input training data (clean)
    xtrain = np.linspace(-10, 10, n_train).reshape(-1, 1)
    ytrain = f(xtrain) + rng.randn(*xtrain.shape) * y_noise
    xtrain_noise = xtrain + x_noise * rng.randn(*xtrain.shape)

    # output testing data (noisy)
    xtest = np.linspace(-10.0 - buffer, 10.0 + buffer, n_test)[:, None]
    ytest = f(xtest)
    xtest_noise = xtest + x_noise * rng.randn(*xtest.shape)

    idx_sorted = np.argsort(xtest_noise, axis=0)
    xtest_noise = xtest_noise[idx_sorted[:, 0]]
    ytest_noise = ytest[idx_sorted[:, 0]]

    return xtrain, xtrain_noise, ytrain, xtest, xtest_noise, ytest, ytest_noise

n_train = 10
n_test = 10
x_noise = 0.3
y_noise = 0.05
seed = 123


(
    Xtrain,
    Xtrain_noise,
    ytrain,
    xtest,
    xtest_noise,
    ytest,
    ytest_noise,
) = regression_near_square(
    n_train=n_train, n_test=n_test, x_noise=x_noise, y_noise=0.05, seed=123, buffer=0.3
)

x_stddev = np.array([x_noise])

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(Xtrain_noise, ytrain, color="tab:orange", label="Training Data")
ax.plot(xtest, ytest, color="black", label="True Function")
ax.legend()
plt.tight_layout()
# plt.show()



xtrain_tensor = torch.Tensor(Xtrain_noise)
ytrain_tensor = torch.Tensor(ytrain.squeeze())
xtest_tensor = torch.Tensor(xtest_noise)
ytest_tensor = torch.Tensor(ytest_noise)

print(f"Shape of xtrain_tensor: {xtrain_tensor.shape}")

def plot_predictions(mu, lower, upper, noisy=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    if noisy:
        ax.scatter(xtest_noise, ytest_noise, marker="o", s=30, color="tab:orange", label="Noisy Test Data")
    else:
        ax.scatter(xtest, ytest, marker="o", s=30, color="tab:orange", label="Noisy Test Data")
    ax.plot(xtest, ytest, color="black", linestyle="-", label="True Function")
    ax.plot(
        xtest,
        mu.ravel(),
        color="Blue",
        linestyle="--",
        linewidth=3,
        label="Predictive Mean",
    )
    ax.fill_between(
        xtest.ravel(),
        lower,
        upper,
        alpha=0.4,
        color="tab:blue",
        label=f" 95% Confidence Interval",
    )
    ax.plot(xtest, lower, linestyle="--", color="tab:blue")
    ax.plot(xtest, upper, linestyle="--", color="tab:blue")
    plt.tight_layout()
    plt.legend(fontsize=12)
    return fig, ax


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(xtrain_tensor, ytrain_tensor, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
losses = []
training_iter = 250
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * training_iter], gamma=0.1)

with tqdm.trange(training_iter) as pbar:
    for i in pbar:
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(xtrain_tensor)
        # Calc loss and backprop gradients
        loss = -mll(output, ytrain_tensor)
        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(losses)
ax.set(title="Loss", xlabel="Iterations", ylabel="Negative Log-Likelihood")
plt.show()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()


# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(xtest_tensor))


mu = observed_pred.mean.detach().numpy()
# get variance
var = observed_pred.variance.detach().numpy()
std = np.sqrt(var.squeeze())
# Get upper and lower confidence bounds
lower, upper = observed_pred.confidence_region()
lower, upper = lower.detach().numpy(), upper.detach().numpy()

plot_predictions(mu, lower, upper)
plt.show()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

X = torch.autograd.Variable(torch.Tensor(xtest_noise), requires_grad=True)
observed_pred = likelihood(model(X))
dydtest_x_ag  = torch.autograd.grad(observed_pred.mean.sum(), X)[0]

fig, ax = plt.subplots(figsize=(16,5))
ax.plot(xtest_tensor.detach().numpy(), mu, color="black", label="Mean")
# ax.plot(xtest_tensor.detach().numpy(), dydtest_x.detach().numpy(), 'green', label="Gradient (Tensor)")
ax.plot(xtest_tensor.detach().numpy(), dydtest_x_ag.detach().numpy(), 'orange', linestyle="-.", label="Gradient (Autograd)")
# ax.plot(xtest_tensor.detach().numpy(), dydtest_x_f.detach().numpy(), 'red', linestyle="--", label="Gradient (Functional)")
plt.legend()
plt.show()

# Continue from your existing code

# Ensure the model is in evaluation mode
model.eval()
likelihood.eval()

# Prepare the test inputs for gradient computation
X = torch.autograd.Variable(torch.Tensor(xtest_noise), requires_grad=True)

# Initialize an empty list to store the Jacobian rows
jacobian = []

# Compute the Jacobian
for i in range(X.shape[0]):
    # Zero out previous gradients
    model.zero_grad()
    likelihood.zero_grad()

    # Get the prediction for the i-th test input
    observed_pred_i = likelihood(model(X[i].unsqueeze(0)))

    # Compute gradients with respect to the i-th input
    grad_i = torch.autograd.grad(observed_pred_i.mean, X)[0][i]

    # Append the computed gradients (partial derivatives) to the Jacobian list
    jacobian.append(grad_i)

# Convert the list of gradients to a tensor (Jacobian matrix)
jacobian_matrix = torch.stack(jacobian)

print("Jacobian matrix size:", jacobian_matrix.size())
print("Jacobian matrix:", jacobian_matrix)
