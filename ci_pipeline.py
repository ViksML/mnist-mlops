import subprocess
import sys
import unittest
import torch
from test_basic import TestMNISTModel

def run_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMNISTModel))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def check_dependencies():
    """Verify all required packages are installed"""
    required_packages = [
        'torch',
        'torchvision',
        'matplotlib',
        'numpy'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            return False
    return True

def verify_cuda():
    """Check CUDA availability"""
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
    return True

def main():
    """Main CI pipeline"""
    print("Starting CI Pipeline...")
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Verify CUDA
    print("\nChecking CUDA...")
    verify_cuda()
    
    # Run tests
    print("\nRunning tests...")
    if not run_tests():
        sys.exit(1)
    
    print("\nCI Pipeline completed successfully!")

if __name__ == "__main__":
    main() 