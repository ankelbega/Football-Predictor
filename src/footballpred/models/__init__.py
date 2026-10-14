"""Model-stage implementations that turn causal features into forecast probabilities.

Baselines, Elo, Dixon–Coles and LightGBM share a small interface so evaluation can compare
their outputs using the same splits and proper scoring rules.
"""
