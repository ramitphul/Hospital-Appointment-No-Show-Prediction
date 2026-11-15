"""Explainable AI module."""
from src.explainability.shap_explainer import ModelExplainer, generate_global_shap_explanations

__all__ = ["ModelExplainer", "generate_global_shap_explanations"]
