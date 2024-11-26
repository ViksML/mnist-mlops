import unittest
import torch
import torch.nn as nn
from model import LightweightCNN, count_parameters
from train import load_data, train_model, test
import numpy as np
from torchvision import transforms

class TestAdvancedMNISTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = LightweightCNN().to(cls.device)
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = torch.optim.Adam(cls.model.parameters(), lr=0.001)
    
    def test_gradient_flow(self):
        """Test if gradients are flowing properly through all layers"""
        self.model.train()
        
        # Create dummy batch
        batch_size = 8
        dummy_input = torch.randn(batch_size, 1, 28, 28).to(self.device)
        dummy_target = torch.randint(0, 10, (batch_size,)).to(self.device)
        
        # Forward and backward pass
        self.optimizer.zero_grad()
        output = self.model(dummy_input)
        loss = self.criterion(output, dummy_target)
        loss.backward()
        
        # Check gradients for each layer
        gradients_present = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                has_gradient = param.grad is not None and torch.sum(torch.abs(param.grad)) > 0
                gradients_present.append((name, has_gradient))
                self.assertTrue(has_gradient, f"No gradient flow in layer: {name}")
        
        # Ensure all layers have gradients
        self.assertTrue(all(grad for _, grad in gradients_present), 
                       "Some layers are not receiving gradients")
    
    def test_model_robustness(self):
        """Test model's robustness to input noise"""
        self.model.eval()
        
        # Create a clean test input
        test_input = torch.zeros(1, 1, 28, 28).to(self.device)
        test_input[0, 0, 14:18, 14:18] = 1.0  # Create a simple square pattern
        
        # Test with different noise levels
        noise_levels = [0.1, 0.2, 0.3]
        predictions = []
        
        for noise_level in noise_levels:
            noisy_input = test_input + noise_level * torch.randn_like(test_input)
            with torch.no_grad():
                output = self.model(noisy_input)
                pred = torch.argmax(output, dim=1)
                predictions.append(pred.item())
        
        # Check if predictions are consistent under moderate noise
        unique_predictions = len(set(predictions))
        self.assertLessEqual(unique_predictions, 2, 
                            "Model predictions are too volatile under noise")
    
    def test_activation_statistics(self):
        """Test if activations have good statistical properties"""
        self.model.eval()
        activation_values = []
        
        # Hook to collect activation values
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_values.append(output.detach().cpu().numpy())
        
        # Register hooks for ReLU layers
        hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass with random input
        test_input = torch.randn(32, 1, 28, 28).to(self.device)
        with torch.no_grad():
            _ = self.model(test_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze activation statistics
        for i, activations in enumerate(activation_values):
            active_neurons = np.mean(activations > 0)
            self.assertGreater(active_neurons, 0.05, 
                             f"Layer {i} has too few active neurons")
            self.assertLess(active_neurons, 0.95, 
                           f"Layer {i} has too many active neurons")

if __name__ == '__main__':
    unittest.main() 