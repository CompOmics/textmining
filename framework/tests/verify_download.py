
import os
import requests
import unittest
from unittest.mock import MagicMock, patch
from normalization.download import download_ontology

class TestDownloadRobustness(unittest.TestCase):
    
    @patch('normalization.download.requests.Session')
    def test_small_file_rejection(self, mock_session):
        """Test that files smaller than 10KB are rejected (simulating error page)."""
        print("\n=== Testing Small File Rejection ===")
        
        # Mock response to be < 10KB
        mock_response = MagicMock()
        mock_response.headers.get.return_value = '100' # 100 bytes
        mock_response.iter_content.return_value = [b'x' * 100]
        mock_response.raise_for_status.return_value = None
        
        # Setup session mock
        mock_sess_inst = MagicMock()
        mock_sess_inst.get.return_value = mock_response
        mock_session.return_value = mock_sess_inst
        
        # Run download
        success, msg = download_ontology('cl', output_dir=None, force=True)
        
        print(f"Result: {success}, {msg}")
        
        # Assertions
        self.assertFalse(success)
        self.assertIn("too small", msg)
        
        # Verify file was cleaned up
        import pathlib
        path = pathlib.Path("ontologies/cl.obo")
        self.assertFalse(path.exists(), "Small file should be deleted")
        print("✓ Small file rejection passed")

if __name__ == '__main__':
    unittest.main()
