
import os
import yaml
from pathlib import Path
from normalization.config import NormalizationConfig

def test_config_loading():
    print("\n=== Testing Config Loading ===")
    
    # 1. Verify config.yaml exists
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml not found")
        return False
        
    # 2. Load it manually to check expected values
    with open("config.yaml") as f:
        raw_config = yaml.safe_load(f)
        
    expected_backend = raw_config['normalization']['backend']
    expected_gpu = raw_config['normalization']['use_gpu']
    
    # 3. Initialize NormalizationConfig (should auto-load)
    config = NormalizationConfig()
    
    print(f"Config Backend: {config.index_backend} (Expected: {expected_backend})")
    print(f"Config GPU: {config.use_gpu} (Expected: {expected_gpu})")
    
    if config.index_backend == expected_backend and config.use_gpu == expected_gpu:
        print("✓ NormalizationConfig loaded config.yaml correctly")
    else:
        print("❌ NormalizationConfig failed to load overrides")
        return False

    # 4. Check main.py defaults (inspect string)
    # We can't easily import main without running it, but we can check if it runs without errors
    # and help output contains default paths
    import subprocess
    result = subprocess.run(["python3", "main.py", "--help"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ main.py --help failed: {result.stderr}")
        return False
        
    if "./docs" in result.stdout:
        print("✓ main.py --help shows updated defaults (./docs)")
    else:
        print(f"❌ main.py --help missing expected defaults (./docs). Output snippet: {result.stdout[:200]}...")
        return False
        
    return True

if __name__ == "__main__":
    try:
        if test_config_loading():
            print("\nAll config tests passed!")
        else:
            print("\nConfig tests failed.")
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
