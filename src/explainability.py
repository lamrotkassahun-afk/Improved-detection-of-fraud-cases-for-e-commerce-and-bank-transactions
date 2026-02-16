import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class FraudExplainer:
    def __init__(self, model, training_data):
        """
        Initializes the SHAP explainer.
        :param model: The trained model artifact (e.g., RandomForest or XGBoost).
        :param training_data: A sample of processed data used as a background reference for SHAP.
        """
        self.model = model
        # Use TreeExplainer for tree-based ensemble models as per Task 2 requirements [cite: 156, 172]
        self.explainer = shap.TreeExplainer(model)
        self.training_data = training_data

    def get_feature_importance_plot(self):
        """
        Generates a global summary plot of feature importance[cite: 173].
        This addresses Task 3.2 by visualizing what drives fraud detection globally.
        """
        shap_values = self.explainer.shap_values(self.training_data)
        
        # Handle different SHAP output formats (List for RF, Array for XGBoost)
        if isinstance(shap_values, list):
            # For binary classification list, index 1 is usually the 'Fraud' class [cite: 52]
            values_to_plot = shap_values[1]
        else:
            values_to_plot = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(values_to_plot, self.training_data, show=False)
        plt.tight_layout()
        return plt.gcf()

    def explain_prediction(self, instance):
        """
        Provides a SHAP force plot for an individual transaction[cite: 174].
        Fixes the IndexError by checking the shape of the model's output.
        """
        shap_values = self.explainer.shap_values(instance)
        
        # Polymorphic handling of SHAP value indices
        if isinstance(shap_values, list):
            # Format typically returned by sklearn RandomForestClassifier
            # We take index 1 (Fraudulent class) [cite: 52, 175]
            values_to_plot = shap_values[1][0, :]
            expected_value = self.explainer.expected_value[1]
        else:
            # Format typically returned by XGBoost or LightGBM
            # Some versions return a 3D array (samples, features, classes) or 2D array
            if len(shap_values.shape) == 3:
                values_to_plot = shap_values[0, :, 1]
                expected_value = self.explainer.expected_value[1]
            else:
                values_to_plot = shap_values[0, :]
                expected_value = self.explainer.expected_value

        # Generate the Force Plot as required by Task 3 instructions [cite: 174]
        # This helps stakeholders understand the "Reason Codes" for specific flags [cite: 63]
        shap.initjs()
        plot = shap.force_plot(
            expected_value, 
            values_to_plot, 
            instance.iloc[0, :], 
            matplotlib=True, 
            show=False
        )
        return plt.gcf()