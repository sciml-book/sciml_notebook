import os
import sys
import time
import warnings
from typing import Tuple, List, Optional
from functools import partial

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from pyDOE import lhs

class NeuralNetwork(nn.Module):
    """Neural network model for the PINNs implementation."""
    
    def __init__(self, input_size: int, output_size: int, hidden_layers: int, 
                 hidden_units: int, activation_function: nn.Module):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            hidden_layers: Number of hidden layers
            hidden_units: Number of units in each hidden layer
            activation_function: Activation function to use
        """
        super().__init__()
        self.linear_in = nn.Linear(input_size, hidden_units)
        self.linear_out = nn.Linear(hidden_units, output_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_units, hidden_units) for _ in range(hidden_layers)
        ])
        self.act = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        return self.linear_out(x)

class DiscreteTimePINNs:
    """Implementation of Physics-Informed Neural Networks for discrete time problems."""
    
    def __init__(self, config: dict):
        """
        Initialize the DiscreteTimePINNs solver.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_directories()
        self.setup_data()
        self.setup_model()
        self.results = []
        self.iter = 0

    def setup_directories(self):
        """Create necessary directories for output."""
        directories = ['results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def setup_data(self):
        """Load and prepare data for training."""
        # Load Burgers equation data
        data = scipy.io.loadmat(self.config['data_path'])
        self.t = data['t'].flatten()[:, None]
        self.x = data['x'].flatten()[:, None]
        self.Exact = np.real(data['usol']).T.astype(np.float32)
        
        # Setup time steps and initial conditions
        self.setup_time_steps()
        self.setup_initial_conditions()
        self.setup_irk_weights()
        
        # Convert data to tensors
        self.prepare_tensors()

    def setup_time_steps(self):
        """Setup time step information."""
        self.idx_t0 = self.config['idx_t0']
        self.idx_t1 = self.config['idx_t1']
        dt_value = self.t[self.idx_t1] - self.t[self.idx_t0]
        self.dt = torch.tensor(dt_value, dtype=torch.float32)

    def setup_initial_conditions(self):
        """Setup initial and boundary conditions."""
        N = self.config['N']
        idx_x = np.random.choice(self.Exact.shape[1], N, replace=False)
        self.x0 = self.x[idx_x, :]
        self.u0 = self.Exact[self.idx_t0:self.idx_t0+1, idx_x].T
        
        # Add noise if specified
        if self.config.get('noise_u0', 0.0) > 0:
            self.u0 += (self.config['noise_u0'] * np.std(self.u0) * 
                       np.random.randn(*self.u0.shape))
        
        # Setup boundary conditions
        self.x1 = np.vstack((self.config['lb'], self.config['ub']))
        self.x_star = self.x

    def setup_irk_weights(self):
        """Load and setup IRK weights."""
        tmp = np.loadtxt(
            f"./IRK_weights/Butcher_IRK{self.config['q']}.txt", 
            ndmin=2
        ).astype(np.float32)
        weights = np.reshape(tmp[0:self.config['q']**2 + self.config['q']], 
                           (self.config['q']+1, self.config['q']))
        self.IRK_weights = torch.tensor(weights, dtype=torch.float32).T

    def prepare_tensors(self):
        """Convert numpy arrays to PyTorch tensors."""
        # Dictionary of numpy arrays that need to be converted to tensors
        numpy_arrays = {
            'x0': self.x0,
            'x1': self.x1,
            'u0': self.u0,
            'x_star': self.x_star
        }
        
        # Convert numpy arrays to tensors
        for name, array in numpy_arrays.items():
            tensor = torch.from_numpy(array).float().to(self.device)
            if name in ['x0', 'x1', 'x_star']:
                tensor.requires_grad = True
            setattr(self, name, tensor)
            
        # Handle dt and IRK_weights separately as they're already tensors
        self.dt = self.dt.to(self.device)
        self.IRK_weights = self.IRK_weights.to(self.device)

    def setup_model(self):
        """Initialize the neural network model."""
        self.model = NeuralNetwork(
            input_size=1,
            output_size=self.config['q'] + 1,
            hidden_layers=4,
            hidden_units=50,
            activation_function=nn.Tanh()
        ).float().to(self.device)
        
        self.model.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """Initialize network weights using Xavier initialization."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.0)

    @staticmethod
    def set_seed(seed: int = 42):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        np.random.seed(seed)

    def f(self, x: torch.Tensor, x_1: torch.Tensor, 
          dt: torch.Tensor, IRK_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the physics-informed neural network function.
        
        Args:
            x: Input tensor
            x_1: Boundary conditions tensor
            dt: Time step
            IRK_weights: IRK weights tensor
            
        Returns:
            Tuple of tensors (U0, U1)
        """
        nu = 0.01/torch.pi
        U1 = self.model(x)
        U = U1[:, :-1]
        
        # Compute derivatives
        U_x = self.fwd_gradients_0(U, x)
        U_xx = self.fwd_gradients_0(U_x, x)
        
        # Compute physics terms
        F = -U*U_x + nu*U_xx
        U0 = U1 - dt * torch.matmul(F, IRK_weights)
        U1 = self.model(x_1)
        
        return U0, U1

    def compute_loss(self, x: torch.Tensor, x_1: torch.Tensor, 
                    dt: torch.Tensor, IRK_weights: torch.Tensor, 
                    U0_real: torch.Tensor) -> torch.Tensor:
        """Compute the total loss for training."""
        U0, U1 = self.f(x, x_1, dt, IRK_weights)
        return torch.sum((U0_real - U0) ** 2) + torch.sum(U1 ** 2)

    def closure(self, optimizer: torch.optim.Optimizer, x: torch.Tensor, 
                x_1: torch.Tensor, x_star: torch.Tensor, dt: torch.Tensor, 
                IRK_weights: torch.Tensor, U0_real: torch.Tensor) -> torch.Tensor:
        """Closure function for LBFGS optimizer."""
        optimizer.zero_grad()
        loss = self.compute_loss(x, x_1, dt, IRK_weights, U0_real)
        loss.backward(retain_graph=True)
        
        # Update iteration count and compute error
        self.iter += 1
        U1_pred = self.model(x_star)
        pred = U1_pred[:, -1].detach().cpu().numpy()
        error = np.linalg.norm(pred - self.Exact[self.idx_t1, :], 2) / \
                np.linalg.norm(self.Exact[self.idx_t1, :], 2)
        
        # Store results and save model if needed
        self.results.append([self.iter, loss.item(), error])    
        return loss

    def train(self, num_iter: int):
        """Train using L-BFGS optimizer."""
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=num_iter,
            max_eval=num_iter,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=100,
            line_search_fn='strong_wolfe'
        )
        
        start_time = time.time()
        closure_fn = partial(self.closure, optimizer, self.x0, self.x1, 
                            self.x_star, self.dt, self.IRK_weights, self.u0)
        
        with tqdm(total=num_iter, desc="LBFGS Training", unit="iter") as pbar:
            def closure_with_tqdm():
                loss = closure_fn()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                return loss

            optimizer.step(closure_with_tqdm)
            
        self.lbfgs_training_time = time.time() - start_time
        # Save final results
        self.save_results()


    def save_results(self):
        """Save training results and model."""
        # Save training summary
        total_time = getattr(self, 'lbfgs_training_time', 0)
        
        with open('results/burgers_discrete_time_training_summary.txt', 'w') as f:
            f.write(f"Total training time: {total_time:.6e} seconds\n")
            f.write(f"Total iterations: {self.iter}\n")
            f.write(f"Final Loss: {self.results[-1][1]:.6e}\n")
            f.write(f"Final L2: {self.results[-1][2]:.6e}\n")
        
        # Save training data
        results_array = np.array(self.results)
        np.savetxt("results/burgers_discrete_time_training_data.csv", 
                   results_array, 
                   delimiter=",", 
                   header="Iter,Loss,L2",
                   comments="")
        
        # Save final model
        self.save_model('burgers_discrete_time.pt')

    def save_model(self, path: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def plot_results(self):
        """Generate plots for the results."""
        # Create figure for training curves
        self.plot_training_curves()
        
        # Create figure for solution visualization
        self.plot_solution()

    def plot_training_curves(self):
        """Plot training loss and L2 error curves."""
        data = pd.read_csv('results/burgers_discrete_time_training_data.csv')
        
        fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axarr[0].semilogy(data['Iter'], data['Loss'], 
                        label='Loss', color='gray', linewidth=1)
        axarr[0].set_xlabel('Iteration')
        axarr[0].set_ylabel('Loss')
        
        # L2 error plot
        axarr[1].semilogy(data['Iter'], data['L2'], 
                        label='L2 Error', color='gray', linewidth=1)
        axarr[1].set_xlabel('Iteration')
        axarr[1].set_ylabel(r'$\mathrm{L}_2$')
        
        plt.tight_layout()
        plt.savefig('results/burgers_discrete_time_training_curves.pdf')
        plt.close()

    def plot_solution(self):
        """Plot the solution, including exact solution and predictions."""
        fig = plt.figure(figsize=(12, 10))
        fsize = 12
        # Set up GridSpec
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)
        
        # Plot exact solution
        ax = plt.subplot(gs0[:, :])
        h = ax.imshow(self.Exact.T, interpolation='nearest', cmap='rainbow',
                        extent=[self.t.min(), self.t.max(), self.x.min(), self.x.max()],
                        origin='lower', aspect='auto')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
        
        # Add time slice indicators
        line = np.linspace(self.x.min(), self.x.max(), 2)[:, None]
        # Fix scalar conversion by using item()
        t0_scalar = self.t[self.idx_t0].item()
        t1_scalar = self.t[self.idx_t1].item()
        ax.plot(t0_scalar * np.ones((2,1)), line, 'w-', linewidth=1)
        ax.plot(t1_scalar * np.ones((2,1)), line, 'w-', linewidth=1)
        
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('$u(t,x)$', fontsize=fsize)
        
        # Plot initial condition
        ax = plt.subplot(gs1[0, 0])
        ax.plot(self.x, self.Exact[self.idx_t0,:], 'b-', linewidth=2)
        ax.plot(self.x0.cpu().detach().numpy(), self.u0, 'rx', linewidth=2, label='Data')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = %.2f$' % t0_scalar, fontsize=fsize)
        ax.set_xlim([self.config['lb']-0.1, self.config['ub']+0.1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.3, -0.25), ncol=2, frameon=False)
        
        # Plot prediction vs exact solution
        ax = plt.subplot(gs1[0, 1])
        U1_pred = self.model(self.x_star)
        U1_pred = U1_pred.cpu().detach().numpy()
        
        ax.plot(self.x, self.Exact[self.idx_t1,:], 'b-', linewidth=2, label='Exact')
        ax.plot(self.x_star.cpu().detach().numpy(), U1_pred[:,-1], 'r--', 
                linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = %.2f$' % t1_scalar, fontsize=fsize)
        ax.set_xlim([self.config['lb']-0.1, self.config['ub']+0.1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
        
        plt.savefig('results/burgers_discrete_time.pdf')
        plt.show()
        plt.close()

    @staticmethod
    def fwd_gradients_0(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute forward gradients."""
        z = torch.ones(dy.shape, device=dy.device).requires_grad_()
        g = torch.autograd.grad(dy, x, grad_outputs=z, create_graph=True)[0]
        ones = torch.ones(g.shape, device=g.device)
        return torch.autograd.grad(g, z, grad_outputs=ones, create_graph=True)[0]

def main():
    """Main function to run the DiscreteTimePINNs solver."""
    # Configuration dictionary
    config = {
        'data_path': './burgers_shock.mat',
        'q': 500,
        'N': 250,
        'lb': np.array([-1.0]),
        'ub': np.array([1.0]),
        'idx_t0': 10,
        'idx_t1': 90,
        'noise_u0': 0.0
    }
    
    # Set random seed
    DiscreteTimePINNs.set_seed(42)
    
    # Initialize solver
    solver = DiscreteTimePINNs(config)
    
    # Train the model
    solver.train(num_iter=10000)
    
    # Plot results
    solver.plot_results()
    
    # Print final error
    U1_pred = solver.model(solver.x_star)
    U1_pred = U1_pred.cpu().detach().numpy()
    error = np.linalg.norm(U1_pred[:,-1] - solver.Exact[solver.idx_t1,:], 2) / \
            np.linalg.norm(solver.Exact[solver.idx_t1,:], 2)
    print('Final Error: %e' % (error))

if __name__ == "__main__":
    main()