
import unittest
from unittest.mock import MagicMock, patch
import time
import shutil
from pathlib import Path
from core.extractor import BaseExtractor

class TestExtractor(BaseExtractor):
    def get_prompt(self, text: str) -> str:
        return f"Process {text}"

class TestBatchProcessing(unittest.TestCase):
    def setUp(self):
        self.input_dir = Path("tests/batch_input")
        self.output_dir = Path("tests/batch_output")
        self.input_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create 4 dummy files
        for i in range(4):
            (self.input_dir / f"doc_{i}.txt").write_text(f"Dummy text {i}")

    def tearDown(self):
        if self.input_dir.exists():
            shutil.rmtree(self.input_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @patch('core.extractor.LLMClient')
    def test_concurrency_speedup(self, MockLLMClient):
        # Setup mock to sleep for 1.0 second per call
        mock_llm = MockLLMClient.return_value
        def delayed_response(*args, **kwargs):
            time.sleep(1.0) # Simulate network delay
            return 'THOUGHT PROCESS: Thinking...\nFINAL JSON: {"test": ["val", "evidence"]}'
        mock_llm.get_completion.side_effect = delayed_response

        # Test Sequential (1 worker)
        print("\nTesting Sequential Processing (Workers=1)...")
        start_time = time.time()
        extractor = TestExtractor(self.input_dir, self.ou tput_dir, temperatures=[0.0], max_workers=1)
        # We need to inject the mock into the instance because it's created in __init__
        extractor.llm = mock_llm 
        
        extractor.run()
        seq_duration = time.time() - start_time
        print(f"Sequential duration: {seq_duration:.2f}s")

        # Test Parallel (4 workers)
        print("\nTesting Parallel Processing (Workers=4)...")
        start_time = time.time()
        extractor = TestExtractor(self.input_dir, self.output_dir, temperatures=[0.0], max_workers=4)
        extractor.llm = mock_llm
        
        extractor.run()
        par_duration = time.time() - start_time
        print(f"Parallel duration: {par_duration:.2f}s")
        
        # Assertions
        self.assertGreater(seq_duration, 3.5, "Sequential should take at least 4*delay (approx)")
        self.assertLess(par_duration, 2.5, "Parallel should be significantly faster than sequential")
        print("Success: Parallel execution was faster!")

if __name__ == '__main__':
    unittest.main()
