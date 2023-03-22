from io import StringIO
from unittest.mock import mock_open, patch
import pandas as pd

from ..src.data_utils import import_data

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	# Define the contents of the CSV file to be read
	csv_data = 'name,age\nAlice,30\nBob,40\nCharlie,50\n'

    # Create a mock file object that returns the CSV data when read
	m = mock_open(read_data=csv_data)

	# Use patch to replace the built-in 'open' function with our mock object
	with patch('builtins.open', m):
    	# Call the function with a dummy file path (not used when using a file object)
		df = import_data('dummy_file_path')

    # Define the expected DataFrame
	expected_df = pd.read_csv(StringIO(csv_data), index_col=0)

    # Compare the expected and actual DataFrames
	pd.testing.assert_frame_equal(df, expected_df)