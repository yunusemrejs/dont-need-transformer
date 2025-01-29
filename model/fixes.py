def apply_fixes():
    """Apply common fixes before running tests."""
    import torch
    
    # Fix random seeds
    torch.manual_seed(42)
    
    # Ensure CUDA is available if needed
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        
    # Set default dtype
    torch.set_default_dtype(torch.float32)
    
    # Disable gradient computation for tests
    torch.set_grad_enabled(False)

if __name__ == '__main__':
    apply_fixes()
    from test_wrapper import run_tests_with_debug
    success = run_tests_with_debug()
    print(f"\nTest suite {'passed' if success else 'failed'}")
