# Modified from https://github.com/kevinsbello/dagma/blob/main/src/dagma/nonlinear.py
# Modifications Copyright (C) 2023 Dan Waxman

import copy
import torch
import math
import torch.nn as nn
import numpy as np
from torch import optim
import copy
from tqdm.auto import tqdm
from .locally_connected import LocallyConnected
import abc
import typing
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood

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


class DagmaDCE:
    def __init__(self, model: Dagma_DCE_Module, use_mse_loss=True):
        """Initializes a DAGMA DCE model. Requires a `DAGMA_DCE_Module`

        Args:
            model (Dagma_DCE_Module): module implementing adjacency matrix,
                h_func constraint, and L1 regularization
            use_mse_loss (bool, optional): to use MSE loss instead of log MSE loss.
                Defaults to True.
        """
        self.model = model
        self.loss = self.mse_loss if use_mse_loss else self.log_mse_loss
        self.loss = self.mll_loss if use_mse_loss else self.log_mse_loss

    def mll_loss(self, output, target):

        batched_likelihood, batched_model = self.model.likelihood_list, self.model.model_list
        output = batched_model(*batched_model.train_inputs)
        mll = gpytorch.mlls.SumMarginalLogLikelihood(batched_likelihood, batched_model)
        # print(f"type of output is: {output}")
        # print(f"type of batched_model.train_targets is: {batched_model.train_targets}")
        loss = -mll(output, batched_model.train_targets)
        #print(f"loss is: {loss}")
        return loss

    def mse_loss(self, output: torch.Tensor, target: torch.Tensor):
        """Computes the MSE loss sum (output - target)^2 / (2N)"""
        n, d = target.shape
        # if isinstance(output, torch.distributions.MultivariateNormal):
        #     output_mean = output.mean
        # else:
        #     output_mean = output
        print(f"MSE loss is {0.5 / n * torch.sum((output - target) ** 2)}")
        return 0.5 / n * torch.sum((output - target) ** 2)


    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor):
        """Computes the MSE loss d / 2 * log [sum (output - target)^2 / N ]"""
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def minimize(
        self,
        max_iter: int,
        lr: float,
        lambda1: float,
        lambda2: float,
        mu: float,
        s: float,
        pbar: tqdm,
        lr_decay: bool = False,
        checkpoint: int = 1000,
        tol: float = 1e-3,
    ):
        """Perform minimization using the barrier method optimization

        Args:
            max_iter (int): maximum number of iterations to optimize
            lr (float): learning rate for adam
            lambda1 (float): regularization parameter
            lambda2 (float): weight decay
            mu (float): regularization parameter for barrier method
            s (float): DAMGA constraint hyperparameter
            pbar (tqdm): progress bar to use
            lr_decay (bool, optional): whether or not to use learning rate decay.
                Defaults to False.
            checkpoint (int, optional): how often to checkpoint. Defaults to 1000.
            tol (float, optional): tolerance to terminate learning. Defaults to 1e-3.
        """
        print(f"model parameters are:")
        for param in self.model.parameters():
            print(param)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.99, 0.999),
            weight_decay=mu * lambda2,
        )

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"total params is: {trainable_params}")
        obj_prev = 1e16

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8 if lr_decay else 1.0
        )

        for i in range(max_iter):
            optimizer.zero_grad()

            if i == 0:
                # print(f"self.X is {self.X}")
                # print(f"Shape of self.X is {self.X.shape}")
                X_hat = self.model(self.X)
                # print(f"X_hat is {X_hat}")
                # print(f"Shape of X_hat is {X_hat.shape}")
                score = self.loss(X_hat, self.X)
                # print('afsadgag')
                # print(X_hat.requires_grad)
                # print(f"score is {score}")
                obj = score

            else:
                W_current, observed_derivs = self.model.get_graph(self.X)
                h_val = self.model.h_func(W_current, s)

                if h_val.item() < 0:
                    return False
                X_hat = self.model(self.X)
                # print(X_hat)
                # print(X_hat.requires_grad)
                # X_hat = self.model(self.X).mean  # Extract mean from MultivariateNormal
                # TODO: Use marginal likelihood loss function
                # print(f"type of X_hat is: {type(X_hat)}")
                # print(f"shape of X_hat is: {X_hat.shape}")
                # print(f"X_hat is: {X_hat}")
                # print(f"type of self.X is: {type(self.X)}")
                # print(f"shape of self.X is: {self.X.shape}")
                # print(f"self.X is: {self.X}")
                score = self.mll_loss(X_hat, self.X)

                l1_reg = lambda1 * self.model.get_l1_reg(observed_derivs)
                # print(f"l1_reg is {l1_reg}")
                # print(f"h_val is {h_val}")
                obj = mu * (score + l1_reg) + h_val

                if (i % 500 == 0):
                    print('Objective:', obj, 'mu:', mu, 'score:', score, 'l1_reg:', l1_reg, 'h_val:', h_val)
                    print(W_current, observed_derivs)

            obj.backward()
            optimizer.step()

            if lr_decay and (i + 1) % 1000 == 0:
                scheduler.step()
            
            if i % checkpoint == 0 or i == max_iter - 1:
                obj_new = obj.item()

                if np.abs((obj_prev - obj_new) / (obj_prev)) <= tol:
                    pbar.update(max_iter - i)
                    break
                obj_prev = obj_new

            pbar.update(1)

        return True

    def fit(
        self,
        X: torch.Tensor,
        lambda1: float = 0.02,
        lambda2: float = 1e-4,
        T: int = 4,
        mu_init: float = 1.0,
        mu_factor: float = 0.1,
        s: float = 1.0,
        warm_iter: int = 5e3,
        max_iter: int = 8e3,
        lr: float = 1e-3,
        disable_pbar: bool = False,
    ) -> torch.Tensor:
        """Fits the DAGMA-DCE model

        Args:
            X (torch.Tensor): inputs
            lambda1 (float, optional): regularization parameter. Defaults to 0.02.
            lambda2 (float, optional): weight decay. Defaults to 0.005.
            T (int, optional): number of barrier loops. Defaults to 4.
            mu_init (float, optional): barrier path coefficient. Defaults to 1.0.
            mu_factor (float, optional): decay parameter for mu. Defaults to 0.1.
            s (float, optional): DAGMA constraint hyperparameter. Defaults to 1.0.
            warm_iter (int, optional): number of warmup models. Defaults to 5e3.
            max_iter (int, optional): maximum number of iterations for learning. Defaults to 8e3.
            lr (float, optional): learning rate. Defaults to 1e-3.
            disable_pbar (bool, optional): whether or not to use the progress bar. Defaults to False.

        Returns:
            torch.Tensor: graph returned by the model
        """
        mu = mu_init
        self.X = X

        with tqdm(total=(T - 1) * warm_iter + max_iter, disable=disable_pbar) as pbar:
            for i in range(int(T)):
                success, s_cur = False, s
                lr_decay = False

                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                print(f"success : {success}")
                while success is False:
                    
                    success = self.minimize(
                        inner_iter,
                        lr,
                        lambda1,
                        lambda2,
                        mu,
                        s_cur,
                        lr_decay=lr_decay,
                        pbar=pbar,
                    )

                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5
                        lr_decay = True
                        if lr < 1e-10:
                            print(":(")
                            break  # lr is too small

                    mu *= mu_factor
                    print(f"Success is {success}")

        return self.model.get_graph(self.X)[0]


class DagmaMLP_DCE(Dagma_DCE_Module):
    def __init__(
        self,
        dims: typing.List[int],
        bias: bool = True,
        dtype: torch.dtype = torch.double,
    ):
        """Initializes the DAGMA DCE MLP module

        Args:
            dims (typing.List[int]): dims
            bias (bool, optional): whether or not to use bias. Defaults to True.
            dtype (torch.dtype, optional): dtype to use. Defaults to torch.double.
        """
        torch.set_default_dtype(dtype)

        super(DagmaMLP_DCE, self).__init__()

        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)

        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)

        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))

        self.fc2 = nn.ModuleList(layers)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sigmoidal feedforward NN

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        
        x = self.fc1(x)

        x = x.view(-1, self.dims[0], self.dims[1])

        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)

        x = x.squeeze(dim=2)

        return x

    def get_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Get the adjacency matrix defined by the DCE and the batched Jacobian

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor, torch.Tensor: the weighted graph and batched Jacobian
        """
        x_dummy = x.detach().requires_grad_()

        observed_deriv = torch.func.vmap(torch.func.jacrev(self.forward))(x_dummy).view(
            -1, self.d, self.d
        )
        W = torch.sqrt(torch.mean(observed_deriv**2, axis=0).T)


        return W, observed_deriv

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

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, i_to_delete):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.i_to_delete = i_to_delete
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[-1]-1))

    def forward(self, x):
        x = x[:, np.r_[:self.i_to_delete, self.i_to_delete+1:x.shape[-1]]]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DagmaGP_DCE(Dagma_DCE_Module):
    def __init__(self, train_x, num_tasks, lr=0.1, training_iterations=50):
        super(DagmaGP_DCE, self).__init__()
        self.num_tasks = num_tasks
        self.lr = lr
        self.training_iterations = training_iterations
        self.I = torch.eye(self.num_tasks)
        self.d = num_tasks

        self.models = []
        self.likelihoods = []
        for i in range(self.num_tasks):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x, train_x[:, i], likelihood, i_to_delete=i)
            self.models.append(model)
            self.likelihoods.append(likelihood)

        self.model_list = gpytorch.models.IndependentModelList(*self.models)
        self.likelihood_list = gpytorch.likelihoods.LikelihoodList(*self.likelihoods)

        self.model_list.train()
        self.likelihood_list.train()

    def forward(self, x):

        # for model in self.model_list.models:
        #     model.eval()

        self.model_list.train()
        self.likelihood_list.train()

        predictive_means = []
        predictive_variances = []
        for model, likelihood in zip(self.model_list.models, self.likelihood_list.likelihoods):
            observed_pred = likelihood(model(x))
            predictive_means.append(observed_pred.mean)
            predictive_variances.append(observed_pred.variance)

        return torch.stack(predictive_means, dim = 1)

    # def train(self, train_x):

    #     self.model_list.train()
    #     self.likelihood_list.train()
    #     optimizer = torch.optim.Adam(self.model_list.parameters(), lr=self.lr)

    #     for i in range(self.training_iterations):
    #         optimizer.zero_grad()
    #         output = self.model_list(*[train_x for _ in range(self.num_tasks)])
    #         targets = [train_x[:, j] for j in range(self.num_tasks)]
    #         loss = -self.mll(output, targets)
    #         loss.backward()
    #         optimizer.step()
    #         print('Iter %d/%d - Loss: %.3f' % (i + 1, self.training_iterations, loss.item()))

    def get_graph(self, x):
        # for model in self.model_list.models:
        #     model.eval()

        # self.model_list.eval()
        # self.likelihood_list.eval()

        x_full = x.detach().requires_grad_(True)
        derivative = torch.zeros(len(x), self.num_tasks, self.d)

        for j, model in enumerate(self.model_list.models):
            model.zero_grad()
            observed_pred = self.likelihoods[j](model(x_full))
            mean = observed_pred.mean
            grad_outputs = torch.ones(mean.shape)
            gradients = torch.autograd.grad(outputs=mean.sum(), inputs=x_full)[0]
            derivative[:, j] = gradients

        mean_squared_jacobians = torch.mean(derivative ** 2, dim=0)
        W = torch.sqrt(mean_squared_jacobians)
        return W, derivative


    # def get_graph(self, x):
    #     for model in self.model_list.models:
    #         model.eval()

    #     derivative = torch.zeros(len(x), self.num_tasks, self.d)

    #     batch_size = 1024
    #     num_batches = int(math.ceil(len(x) / batch_size))

    #     for batch_idx in range(num_batches):
    #         start_idx = batch_idx * batch_size
    #         end_idx = min(start_idx + batch_size, len(x))
    #         x_batch = x[start_idx:end_idx].detach().requires_grad_(True)

    #         for j, model in enumerate(self.model_list.models):
    #             model.zero_grad()
    #             observed_pred = self.likelihoods[j](model(x_batch))
    #             mean = observed_pred.mean
    #             grad_outputs = torch.ones(mean.shape)
    #             gradients = torch.autograd.grad(outputs=mean, inputs=x_batch, grad_outputs=grad_outputs, create_graph=True)[0]
    #             derivative[start_idx:end_idx, j] = gradients

    #     mean_squared_jacobians = torch.mean(derivative ** 2, dim=0)
    #     W = torch.sqrt(mean_squared_jacobians)
    #     return W, derivative

    # def get_graph(self, x):
    #     """
    #     Computes the gradient of the output mean with respect to the input for each model in the model_list.
    #     """
    #     for model in self.model_list.models:
    #         model.eval()

    #     W_all = []  # To store the gradient norms for each model

    #     for model, likelihood in zip(self.model_list.models, self.likelihood_list.likelihoods):
    #         with torch.enable_grad():
    #             input_vector = x.detach().requires_grad_(True)
    #             model.zero_grad()
    #             observed_pred = likelihood(model(input_vector))
    #             mean = observed_pred.mean
    #             mean.backward()

    #             gradient = input_vector.grad
    #             W = torch.sqrt(torch.mean(gradient ** 2, axis=0))
    #             W_all.append(W)

    #             input_vector.grad.zero_()

    #     W_all = torch.stack(W_all, dim=1)
    #     print(f"Shape of W_all is: {W_all.shape}")
    #     return W_all


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
        #print(h)
        return h

    def get_l1_reg(self, observed_derivs: torch.Tensor) -> torch.Tensor:
        """Gets the L1 regularization

        Args:
            observed_derivs (torch.Tensor): the batched Jacobian matrix

        Returns:
            torch.Tensor: _description_
        """
        return torch.sum(torch.abs(torch.mean(observed_derivs, axis=0)))

# class DagmaGP_DCE(Dagma_DCE_Module):
#     def __init__(self, train_x, train_y, likelihood):
#         super(DagmaGP_DCE, self).__init__()
        
#         self.gp = gpytorch.models.ExactGP(train_x, train_y, likelihood)
#         self.gp.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([10]))
#         self.gp.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([10]))
#         )
#         self.gp.covar_module.base_kernel.lengthscale = 1.0
#         n, d = train_x.shape
#         self.d = d
#         self.n = n
#         self.I = torch.eye(self.d)
        

#     def forward(self, x):
#         mean_x = self.gp.mean_module(x)
#         covar_x = self.gp.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
#             gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#         )
    
#     def get_graph(self, x):
#         x_dummy = x.detach().requires_grad_()
#         n, d = x_dummy.shape
        
#         lengthscale = self.gp.covar_module.base_kernel.lengthscale
#         print(f"Shape of lengthscale is: {lengthscale.shape}")
#         # print(f"lengthscale is: {lengthscale}")
#         lengthscale_rep = lengthscale.repeat(n//d, 1, 1)
#         # print(f"lengthscale_rep is: {lengthscale_rep}")
        
#         x1 = x_dummy.unsqueeze(2)  # Shape: [n, d, 1]
#         x2 = x_dummy.unsqueeze(1)  # Shape: [n, 1, d]
#         # print(f"x1 and x2 are : {x1} and {x2}")
#         diff = x1 - x2  # Shape: [n, d, d]
#         # print(f"diff is   : {diff}")
#         scaled_diff = diff.pow(2)
#         # scaled_diff = diff.pow(2).sum(dim=-1)
#         print(f"Shape of scaled_diff is: {scaled_diff.shape}")
#         rbf_matrix = torch.exp(-0.5 * scaled_diff)  # shape: [n, d, d]
#         # print(f"rbf_matrix is: {rbf_matrix}")
#         # rbf_derivative = -rbf_matrix.unsqueeze(-1) * (diff)
#         rbf_derivative = -rbf_matrix * (diff)
#         W = torch.sqrt(torch.mean(rbf_derivative**2, axis=0).T) # Shape: [d, d]
#         return W, rbf_derivative

#     def h_func(self, W: torch.Tensor, s: float = 1.0) -> torch.Tensor:
#         """Calculate the DAGMA constraint function

#         Args:
#             W (torch.Tensor): adjacency matrix
#             s (float, optional): hyperparameter for the DAGMA constraint,
#                 can be any positive number. Defaults to 1.0.

#         Returns:
#             torch.Tensor: constraint
#         """

#         h = -torch.slogdet(s * self.I - W * W)[1] + self.d * np.log(s)

#         return h

#     def get_l1_reg(self, observed_derivs: torch.Tensor) -> torch.Tensor:
#         """Gets the L1 regularization

#         Args:
#             observed_derivs (torch.Tensor): the batched Jacobian matrix

#         Returns:
#             torch.Tensor: _description_
#         """
#         return torch.sum(torch.abs(torch.mean(observed_derivs, axis=0)))
    

#     # def __init__(self, train_x, train_y, likelihood, kernel=None):
#     #     """
#     #     Initializes the DagmaGP_DCE module

#     #     Args:
#     #         input_dim (int): The number of input dimensions.
#     #         likelihood (gpytorch.likelihoods.Likelihood): GP likelihood function.
#     #         kernel (gpytorch.kernels.Kernel, optional): GP kernel. If None, a default kernel is used.
#     #     """
#     #     super(DagmaGP_DCE, self).__init__()

#     #     # check batch size args for multioutput GP
#     #     self.gp = gpytorch.models.ExactGP(train_x, train_y, likelihood)
#     #     # check approximate GPs

#     #     if kernel is None:
#     #         self.gp.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#     #     else:
#     #         self.gp.covar_module = kernel

#     #     self.gp.mean_module = gpytorch.means.ConstantMean()

#     # def forward(self, x):
#     #     mean_x = self.gp.mean_module(x)
#     #     covar_x = self.gp.covar_module(x)
#     #     return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    
#     # def get_graph_common(self, x):
#     #     """
#     #     Get the adjaceny matrix defined by the DCE and the batched Jacobians of GP

#     #     Args:
#     #         x (torch.Tensor): input

#     #     Returns:
#     #         torch.Tensor, torch.Tensor: the weighted graph and batched Jacobian
#     #     """
#     #     # same as MLP
#     #     # x = x.requires_grad_(True)
#     #     x_dummy = x.detach().requires_grad_()

#     #     # self.forward from MLP
#     #     with gpytorch.settings.fast_pred_var():
#     #         pred = self(x_dummy)

#     #     # mean derivative
#     #     mean = pred.mean
#     #     mean_derivatives = torch.autograd.grad(outputs=mean, inputs=x_dummy,
#     #                                         grad_outputs=torch.ones_like(mean),
#     #                                         create_graph=True, retain_graph=True, only_inputs=True)[0]

#     #     # cov. jacobians
#     #     covar_matrix = self.gp.covar_module(x_dummy).evaluate()
#     #     covar_jacobian = torch.zeros(*x_dummy.shape[:-1], *covar_matrix.shape)

#     #     for i in range(x_dummy.size(0)):
#     #         grad_outputs = torch.zeros_like(covar_matrix)
#     #         grad_outputs[:, i] = 1  # i-th input
#     #         covar_jacobian[i] = torch.autograd.grad(outputs=covar_matrix, inputs=x,
#     #                                                 grad_outputs=grad_outputs,
#     #                                                 retain_graph=True,
#     #                                                 create_graph=True,
#     #                                                 only_inputs=True)[0]

#     #     # batched derivatives
#     #     combined_derivatives = torch.cat([mean_derivatives.unsqueeze(0), covar_jacobian], dim=0)

#     #     # RMS calculation for the adjacency matrix
#     #     W = torch.sqrt(torch.mean(combined_derivatives**2, dim=(0, 1)))

#     #     return W, combined_derivatives