import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- Model Classes ---

class LIFActivation(nn.Module):
    def __init__(self, threshold=1.0, leak=0.95):
        super(LIFActivation, self).__init__()
        self.threshold = threshold
        self.leak = leak
        self.potential = None

    def forward(self, potential_update):
        if self.potential is None or self.potential.shape != potential_update.shape:
            self.potential = torch.zeros_like(potential_update)

        self.potential = self.potential * self.leak + potential_update
        spike = (self.potential >= self.threshold).float()
        self.potential = torch.where(spike > 0, torch.zeros_like(self.potential), self.potential)
        return spike  # Only return spike for consistency

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class PredictiveCodingLayer(nn.Module):
    def __init__(self, input_dim, units, recurrent_steps=3, local_lr=0.0005, weight_decay=0.0001, num_levels=2):
        super(PredictiveCodingLayer, self).__init__()
        self.units = units
        self.recurrent_steps = recurrent_steps
        self.local_lr = local_lr
        self.weight_decay = weight_decay
        self.num_levels = num_levels
        self.eps = 1e-6

        self.kernel = nn.Linear(input_dim, units, bias=False)
        self.recurrent_weights = nn.Linear(units, units, bias=False)
        self.feedback_kernels = nn.ModuleList([nn.Linear(units, input_dim, bias=False) for _ in range(num_levels)])
        self.latent_projection = nn.Linear(input_dim, units, bias=False)

        self.lif_activation = LIFActivation()
        self.optimizer = optim.Adam(self.parameters(), lr=local_lr, weight_decay=weight_decay)
        self.mu_history = []

    def forward(self, inputs):
        batch_size = inputs.size(0)
        # Initialize mu in latent space (units)
        mu = torch.zeros((batch_size, self.units), device=inputs.device, requires_grad=True)
        self.mu_history = []  # Clear mu_history at the start of each forward pass

        for _ in range(self.recurrent_steps):
            # Reconstruct input from latent mu
            reconstructed_input = self.feedback_kernels[0](mu)
            epsilon = inputs - reconstructed_input
            pi = 1.0 / (epsilon.detach() ** 2 + self.eps)
            precision_error = epsilon * pi

            projected_error = self.kernel(precision_error)
            d_mu = self.recurrent_weights(projected_error)
            mu = mu + self.local_lr * d_mu
            self.mu_history.append(mu.clone().detach())  # Store a detached copy

        feedback_sum = torch.zeros_like(mu)
        for i in range(self.num_levels):
            # Generate feedback through hierarchical processing
            fb = torch.tanh(self.feedback_kernels[i](mu))
            fb = self.latent_projection(fb)
            fb_error = mu - fb  # Error in latent space
            fb_precision = torch.exp(-torch.log(fb_error ** 2 + self.eps))
            feedback_sum += fb * fb_precision

        total_precision = feedback_sum.sum(dim=-1, keepdim=True) + self.eps
        normalized_feedback = feedback_sum / total_precision

        spike = self.lif_activation(mu + normalized_feedback)
        return spike  # Output shape: (batch_size, units)

    def update_weights(self, inputs, mu_history):
        self.optimizer.zero_grad()
        total_loss = 0
        for mu in mu_history:
            reconstructed_input = self.feedback_kernels[0](mu)
            epsilon = inputs - reconstructed_input
            pi = 1.0 / (epsilon ** 2 + self.eps)
            prediction_error = (epsilon ** 2 * pi).sum()
            complexity_cost = (pi * torch.log(pi + self.eps) - pi).sum()
            free_energy = prediction_error + complexity_cost
            total_loss += free_energy

        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

class SparseHebbianLayer(nn.Module):
    def __init__(self, input_dim, units, alpha=0.1, sparsity=0.2):
        super(SparseHebbianLayer, self).__init__()
        self.units = units
        self.alpha = alpha
        self.sparsity = sparsity
        self.kernel = nn.Linear(input_dim, units)

    def forward(self, inputs):
        activations = self.kernel(inputs)
        k = max(1, int(self.units * self.sparsity))
        top_k_values, _ = torch.topk(activations, k, dim=-1)
        sparse_mask = (activations >= top_k_values[:, -1, None]).float()
        activations = activations * sparse_mask
        return activations # Only return activations

    def hebbian_update(self, pre, post):
        # pre: [batch_size, input_dim], post: [batch_size, output_dim]
        batch_size = pre.size(0)
        # Compute outer product for each sample in the batch
        delta_w = self.alpha * torch.bmm(
            post.unsqueeze(2),  # [batch_size, output_dim, 1]
            pre.unsqueeze(1)    # [batch_size, 1, input_dim]
        ).mean(0)  # Average over batch
        self.kernel.weight.data += delta_w

    def reset_weights(self):
        nn.init.xavier_uniform_(self.kernel.weight)
        if self.kernel.bias is not None:
            nn.init.zeros_(self.kernel.bias)

class NonHebbianLayer(nn.Module):
    def __init__(self, input_dim, units, decay_rate=0.01):
        super(NonHebbianLayer, self).__init__()
        self.units = units
        self.decay_rate = decay_rate
        self.kernel = nn.Linear(input_dim, units)

    def forward(self, inputs):
        activations = self.kernel(inputs)
        return activations

    def non_hebbian_decay(self):
        self.kernel.weight.data *= (1 - self.decay_rate)

class HierarchicalPredictiveCodingLayer(nn.Module):
    def __init__(self, input_dim, units, num_levels=2, recurrent_steps=3):
        super(HierarchicalPredictiveCodingLayer, self).__init__()
        self.units = units
        self.num_levels = num_levels
        self.recurrent_steps = recurrent_steps

        # Calculate level dimensions ensuring proper reduction
        level_dims = []
        current_dim = input_dim
        for i in range(num_levels):
            level_dims.append(current_dim)
            current_dim = current_dim // 2

        # Create layers with explicit dimension handling
        self.levels = nn.ModuleList([
            PredictiveCodingLayer(
                input_dim=level_dims[i],
                units=level_dims[i + 1] if i < len(level_dims) - 1 else units,
                recurrent_steps=recurrent_steps
            ) for i in range(num_levels)
        ])
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        all_spikes = []
        current_input = inputs

        for level in self.levels:
            spikes = level(current_input) # Only spikes are passed
            all_spikes.append(spikes)
            current_input = F.relu(spikes)  # Add activation between levels

        combined_output = torch.stack(all_spikes).mean(dim=0)
        return combined_output  # Only the combined output

    def update_weights(self, inputs, all_mu_histories):
        self.optimizer.zero_grad()
        total_loss = 0

        current_input = inputs
        for i, level in enumerate(self.levels):
            level_loss = level.update_weights(current_input, all_mu_histories[i])  # Pass correct mu_histories
            current_input = all_mu_histories[i][-1].detach()
            total_loss += level_loss
        return total_loss

class ContextualOutputHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ContextualOutputHead, self).__init__()
        hidden_dim = max(input_dim, output_dim * 4)  # Ensure enough capacity
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = LayerNorm(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

def build_model(input_dim, output_dim):
    # Define dimensions for progressive reduction
    pc_dim = 256  # Reduced initial dimension for better stability
    model = nn.Sequential(
        # Initial predictive coding to process input
        PredictiveCodingLayer(input_dim, pc_dim, recurrent_steps=2),  # Reduced steps
        # Sparse Hebbian learning for feature extraction
        SparseHebbianLayer(pc_dim, pc_dim // 2, sparsity=0.3),  # Increased sparsity
        # Non-Hebbian for dimensionality reduction
        NonHebbianLayer(pc_dim // 2, pc_dim // 4, decay_rate=0.005),  # Reduced decay
        # Hierarchical processing with fewer levels
        HierarchicalPredictiveCodingLayer(pc_dim // 4, pc_dim // 8, num_levels=2, recurrent_steps=2),
        # Output head for classification
        ContextualOutputHead(pc_dim // 8, output_dim)
    )
    return model

# --- Dataset and DataLoader ---
def prepare_data(dataset_name='MNIST', batch_size=64):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_dim = 28 * 28
        output_dim = 10

    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_dim = 32 * 32 * 3
        output_dim = 10
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        input_dim = 28 * 28
        output_dim = 10

    elif dataset_name == "synthetic":
        num_samples = 10000
        input_dim = 50
        output_dim = 5
        data = torch.randn(num_samples, input_dim)
        targets = torch.randint(0, output_dim, (num_samples,))
        train_dataset = TensorDataset(data, targets)
        test_dataset = TensorDataset(data, targets)
    else:
        raise ValueError("Unsupported dataset name")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, input_dim, output_dim

# --- Training Loop ---
def train(model, data_loader, epochs=10, device='cpu'):
    model.to(device)
    # Initialize model weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, SparseHebbianLayer):
            module.reset_weights()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)

            # Forward pass through layers
            pc_output = model[0](data)
            shl_output = model[1](pc_output)
            nhl_output = model[2](shl_output)
            hpc_output = model[3](nhl_output)
            final_output = model[4](hpc_output)

            loss = F.cross_entropy(final_output, target)
            total_loss += loss.item()
            batch_count += 1

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Custom updates with gradient clipping
            model[0].update_weights(data, model[0].mu_history)
            model[1].hebbian_update(pc_output.detach(), shl_output.detach())
            model[2].non_hebbian_decay()

            # Collect all mu_histories from all levels of HierarchicalPredictiveCodingLayer
            all_hpc_mu_histories = [level.mu_history for level in model[3].levels]
            model[3].update_weights(nhl_output, all_hpc_mu_histories)

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
        avg_loss = total_loss / batch_count
        losses.append(avg_loss)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
        
        # Update learning rate based on loss
        scheduler.step(avg_loss)
    return losses

# --- Evaluation Function ---
def evaluate(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            pc_output = model[0](data)
            shl_output = model[1](pc_output)
            nhl_output = model[2](shl_output)
            hpc_output = model[3](nhl_output)
            final_output = model[4](hpc_output)
            _, predicted = torch.max(final_output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    # Set device with deterministic settings
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # Training parameters
    dataset_name = 'MNIST'
    batch_size = 256  # Smaller batch size for better stability
    epochs = 15     # More epochs for convergence

    # Prepare data
    print(f"Preparing {dataset_name} dataset...")
    train_loader, test_loader, input_dim, output_dim = prepare_data(dataset_name, batch_size)

    # Build and train model
    print("Building model...")
    model = build_model(input_dim, output_dim)
    model.to(device)

    print("Starting training...")
    losses = train(model, train_loader, epochs, device)
    
    print("\nEvaluating model...")
    accuracy = evaluate(model, test_loader, device)
    
    # Save the model
    print("Saving model...")
    torch.save(model.state_dict(), f"{dataset_name}_model.pth")
    print(f"Model saved as {dataset_name}_model.pth")
