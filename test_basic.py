import unittest
import torch
import torch.nn as nn
from model import LightweightCNN, count_parameters
from train import load_data, train_model, test

class TestMNISTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up model and device once for all tests
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = LightweightCNN().to(cls.device)
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = torch.optim.Adam(cls.model.parameters(), lr=0.001)
        
    def test_parameter_count_requirement(self):
        """Test that model has less than 25000 parameters"""
        param_count = count_parameters(self.model)
        self.assertLess(param_count, 25000, 
                       f"Model has {param_count} parameters, which exceeds the limit of 25,000")
        print(f"\nModel parameter count: {param_count}")
        
    def test_training_accuracy_requirement(self):
        """Test that model achieves >95% accuracy in 1 epoch"""
        train_loader, _ = load_data()
        
        # Train for one epoch
        self.model.train()
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        self.assertGreater(accuracy, 95.0, 
                          f"Model achieved only {accuracy:.2f}% training accuracy, which is below the required 95%")
        print(f"Training accuracy after one epoch: {accuracy:.2f}%")

if __name__ == '__main__':
    unittest.main() 