import pytest
import numpy as np
from src.bias_detection.bias_analyzer import BiasAnalyzer
import matplotlib.pyplot as plt

def test_bias_analyzer_initialization():
    """Test BiasAnalyzer initialization."""
    analyzer = BiasAnalyzer()
    assert isinstance(analyzer.metrics, dict)
    assert len(analyzer.metrics) == 0

def test_demographic_parity_analysis():
    """Test demographic parity analysis."""
    analyzer = BiasAnalyzer()
    
    # Create sample data
    predictions = np.array([1, 1, 0, 1, 0, 0])
    sensitive_features = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    
    # Calculate demographic parity difference
    dp_diff = analyzer.analyze_demographic_parity(predictions, sensitive_features)
    
    # Check if the result is a float
    assert isinstance(dp_diff, float)
    # Check if the result is between 0 and 1
    assert 0 <= dp_diff <= 1

def test_equalized_odds_analysis():
    """Test equalized odds analysis."""
    analyzer = BiasAnalyzer()
    
    # Create sample data
    y_true = np.array([1, 1, 0, 1, 0, 0])
    predictions = np.array([1, 1, 0, 1, 0, 0])
    sensitive_features = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    
    # Calculate equalized odds difference
    eo_diff = analyzer.analyze_equalized_odds(y_true, predictions, sensitive_features)
    
    # Check if the result is a float
    assert isinstance(eo_diff, float)
    # Check if the result is between 0 and 1
    assert 0 <= eo_diff <= 1

def test_confusion_matrix_analysis():
    """Test confusion matrix analysis."""
    analyzer = BiasAnalyzer()
    
    # Create sample data
    y_true = np.array([1, 1, 0, 1, 0, 0])
    predictions = np.array([1, 1, 0, 1, 0, 0])
    sensitive_features = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    
    # Calculate confusion matrices
    confusion_matrices = analyzer.analyze_confusion_matrices(
        y_true, predictions, sensitive_features
    )
    
    # Check if the result is a dictionary
    assert isinstance(confusion_matrices, dict)
    # Check if we have confusion matrices for each group
    assert len(confusion_matrices) == 2
    # Check if each confusion matrix is 2x2
    for cm in confusion_matrices.values():
        assert cm.shape == (2, 2)

def test_plot_bias_metrics():
    """Test bias metrics plotting."""
    analyzer = BiasAnalyzer()
    
    # Add some metrics
    analyzer.metrics = {
        'demographic_parity': 0.3,
        'equalized_odds': 0.4
    }
    
    # Create the plot
    plt = analyzer.plot_bias_metrics()
    
    # Check if we got a matplotlib figure
    assert plt is not None

def test_plot_confusion_matrices():
    """Test confusion matrices plotting."""
    analyzer = BiasAnalyzer()
    
    # Add some confusion matrices
    analyzer.metrics['confusion_matrices'] = {
        'A': np.array([[2, 0], [0, 1]]),
        'B': np.array([[1, 0], [0, 2]])
    }
    
    # Create the plot
    plt = analyzer.plot_confusion_matrices()
    
    # Check if we got a matplotlib figure
    assert plt is not None

def test_plot_confusion_matrices_error():
    """Test error handling in confusion matrices plotting."""
    analyzer = BiasAnalyzer()
    
    # Try to plot without any confusion matrices
    with pytest.raises(ValueError):
        analyzer.plot_confusion_matrices() 