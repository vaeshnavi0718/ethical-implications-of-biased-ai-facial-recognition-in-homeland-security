import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt
import seaborn as sns

class BiasAnalyzer:
    def __init__(self):
        """Initialize the BiasAnalyzer class."""
        self.metrics = {}
        
    def analyze_demographic_parity(self, predictions, sensitive_features):
        """
        Analyze demographic parity in model predictions.
        
        Args:
            predictions (np.array): Model predictions
            sensitive_features (np.array): Sensitive attributes (e.g., gender, race)
            
        Returns:
            float: Demographic parity difference
        """
        dp_diff = demographic_parity_difference(
            y_true=None,  # Not needed for demographic parity
            y_pred=predictions,
            sensitive_features=sensitive_features
        )
        self.metrics['demographic_parity'] = dp_diff
        return dp_diff
    
    def analyze_equalized_odds(self, y_true, predictions, sensitive_features):
        """
        Analyze equalized odds in model predictions.
        
        Args:
            y_true (np.array): True labels
            predictions (np.array): Model predictions
            sensitive_features (np.array): Sensitive attributes
            
        Returns:
            float: Equalized odds difference
        """
        eo_diff = equalized_odds_difference(
            y_true=y_true,
            y_pred=predictions,
            sensitive_features=sensitive_features
        )
        self.metrics['equalized_odds'] = eo_diff
        return eo_diff
    
    def analyze_confusion_matrices(self, y_true, predictions, sensitive_features):
        """
        Analyze confusion matrices across different demographic groups.
        
        Args:
            y_true (np.array): True labels
            predictions (np.array): Model predictions
            sensitive_features (np.array): Sensitive attributes
            
        Returns:
            dict: Confusion matrices for each demographic group
        """
        unique_groups = np.unique(sensitive_features)
        confusion_matrices = {}
        
        for group in unique_groups:
            mask = sensitive_features == group
            cm = confusion_matrix(y_true[mask], predictions[mask])
            confusion_matrices[group] = cm
            
        self.metrics['confusion_matrices'] = confusion_matrices
        return confusion_matrices
    
    def plot_bias_metrics(self):
        """
        Plot the bias metrics as a bar chart.
        """
        metrics = {k: v for k, v in self.metrics.items() 
                  if k in ['demographic_parity', 'equalized_odds']}
        
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Bias Metrics Analysis')
        plt.ylabel('Difference Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt
    
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for each demographic group.
        """
        if 'confusion_matrices' not in self.metrics:
            raise ValueError("No confusion matrices available. Run analyze_confusion_matrices first.")
            
        n_groups = len(self.metrics['confusion_matrices'])
        fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
        
        for (group, cm), ax in zip(self.metrics['confusion_matrices'].items(), axes):
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title(f'Confusion Matrix - Group {group}')
            
        plt.tight_layout()
        return plt 