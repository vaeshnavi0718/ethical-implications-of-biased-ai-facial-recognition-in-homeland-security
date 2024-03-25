import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go

class BiasVisualizer:
    def __init__(self):
        """Initialize the BiasVisualizer class."""
        self.style_config = {
            'style': 'whitegrid',
            'context': 'paper',
            'palette': 'colorblind'
        }
        sns.set_theme(**self.style_config)
    
    def plot_demographic_distribution(self, 
                                    data: pd.DataFrame,
                                    demographic_column: str,
                                    title: str = "Demographic Distribution"):
        """
        Plot the distribution of demographic groups in the dataset.
        
        Args:
            data (pd.DataFrame): Dataset containing demographic information
            demographic_column (str): Column name containing demographic information
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=demographic_column)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt
    
    def plot_performance_by_demographic(self,
                                      performance_data: pd.DataFrame,
                                      metric: str,
                                      demographic_column: str,
                                      title: str = "Performance by Demographic Group"):
        """
        Plot performance metrics across different demographic groups.
        
        Args:
            performance_data (pd.DataFrame): Performance metrics data
            metric (str): Metric to plot
            demographic_column (str): Column containing demographic information
            title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(data=performance_data, x=demographic_column, y=metric)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt
    
    def plot_confusion_matrix_heatmap(self,
                                    confusion_matrix: np.ndarray,
                                    labels: List[str],
                                    title: str = "Confusion Matrix"):
        """
        Plot a confusion matrix heatmap.
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            labels (List[str]): Class labels
            title (str): Plot title
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix,
                   annot=True,
                   fmt='d',
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        return plt
    
    def plot_bias_metrics_comparison(self,
                                   metrics_data: Dict[str, float],
                                   title: str = "Bias Metrics Comparison"):
        """
        Plot comparison of different bias metrics.
        
        Args:
            metrics_data (Dict[str, float]): Dictionary of metric names and values
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        metrics = list(metrics_data.keys())
        values = list(metrics_data.values())
        
        plt.bar(metrics, values)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.ylabel('Score')
        plt.tight_layout()
        return plt
    
    def create_interactive_dashboard(self,
                                   performance_data: pd.DataFrame,
                                   demographic_columns: List[str],
                                   metrics: List[str]):
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            performance_data (pd.DataFrame): Performance metrics data
            demographic_columns (List[str]): List of demographic columns to analyze
            metrics (List[str]): List of metrics to visualize
        """
        # Create a figure with subplots
        fig = go.Figure()
        
        # Add traces for each metric
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=performance_data[demographic_columns[0]],
                y=performance_data[metric],
                text=performance_data[metric].round(3),
                textposition='auto',
            ))
        
        # Update layout
        fig.update_layout(
            title="Interactive Performance Dashboard",
            xaxis_title=demographic_columns[0],
            yaxis_title="Score",
            barmode='group',
            template="plotly_white"
        )
        
        return fig
    
    def plot_error_distribution(self,
                              errors: np.ndarray,
                              demographic_groups: np.ndarray,
                              title: str = "Error Distribution by Demographic Group"):
        """
        Plot the distribution of errors across demographic groups.
        
        Args:
            errors (np.ndarray): Array of error values
            demographic_groups (np.ndarray): Array of demographic group labels
            title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=demographic_groups, y=errors)
        plt.title(title)
        plt.xlabel("Demographic Group")
        plt.ylabel("Error")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt
    
    def plot_roc_curves_by_group(self,
                                fpr_dict: Dict[str, np.ndarray],
                                tpr_dict: Dict[str, np.ndarray],
                                title: str = "ROC Curves by Demographic Group"):
        """
        Plot ROC curves for different demographic groups.
        
        Args:
            fpr_dict (Dict[str, np.ndarray]): Dictionary of false positive rates by group
            tpr_dict (Dict[str, np.ndarray]): Dictionary of true positive rates by group
            title (str): Plot title
        """
        plt.figure(figsize=(10, 8))
        
        for group in fpr_dict.keys():
            plt.plot(fpr_dict[group],
                    tpr_dict[group],
                    label=f'Group: {group}')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        return plt 