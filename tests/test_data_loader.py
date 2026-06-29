import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data_loader import FootballDataLoader

class TestFootballDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = FootballDataLoader("config.yaml")

    @patch('src.data_loader.requests.get')
    def test_download_file_new(self, mock_get):
        # Mock requests.get to return fake content
        mock_response = MagicMock()
        mock_response.content = b"col1,col2\n1,2"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        test_dest = "data/raw/test_download.csv"
        # Ensure test dest does not exist
        if os.path.exists(test_dest):
            os.remove(test_dest)

        self.loader._download_file("http://fakeurl.com/data.csv", test_dest)
        
        # Assert file was created and contents match
        self.assertTrue(os.path.exists(test_dest))
        with open(test_dest, "r") as f:
            content = f.read()
        self.assertEqual(content, "col1,col2\n1,2")

        # Cleanup
        if os.path.exists(test_dest):
            os.remove(test_dest)

    def test_paths_configured(self):
        self.assertIsNotNone(self.loader.results_url)
        self.assertIsNotNone(self.loader.shootouts_url)
        self.assertIsNotNone(self.loader.rankings_url)

if __name__ == '__main__':
    unittest.main()
