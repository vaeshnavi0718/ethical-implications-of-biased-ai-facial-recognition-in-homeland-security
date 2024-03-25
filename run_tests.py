import pytest
import sys
import os

def main():
    """Run the test suite."""
    # Add the project root to the Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the tests
    pytest.main(['-v', 'tests/'])

if __name__ == '__main__':
    main() 