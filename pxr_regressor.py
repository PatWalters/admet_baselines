"""
PXR (Pregnane X Receptor) regressor model for ADMET prediction.

This module provides a Gaussian Process Regressor with Tanimoto kernel for predicting
PXR activation from SMILES strings.
"""

from typing import List, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
import useful_rdkit_utils as uru
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from sklearn.gaussian_process.kernels import Kernel


class TanimotoKernel(Kernel):
    """
    Tanimoto (Jaccard) similarity kernel for use with Gaussian Process Regressor.
    
    This kernel computes the Tanimoto similarity between binary fingerprints,
    which is equivalent to the Jaccard index for sets.
    """
    
    def __init__(self):
        """Initialize the Tanimoto kernel."""
        pass

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, eval_gradient: bool = False) -> np.ndarray:
        """
        Compute the Tanimoto similarity kernel matrix.

        Args:
            X: Input array of shape (n_X, n_features) containing binary fingerprints.
            Y: Optional input array of shape (n_Y, n_features). If None, uses X.
            eval_gradient: If True, also return gradient (not implemented).

        Returns:
            Kernel matrix of shape (n_X, n_Y).
        """
        Y = X if Y is None else Y
        n_X = X.shape[0]
        n_Y = Y.shape[0]
        K = np.empty((n_X, n_Y))
        for i in range(n_X):
            for j in range(n_Y):
                a = X[i]
                b = Y[j]
                c = np.logical_and(a, b).sum()
                d = np.logical_or(a, b).sum()
                K[i, j] = c / d if d != 0 else 0
        if eval_gradient:
            return K, np.empty((n_X, n_Y, 0))
        return K

    def diag(self, X: np.ndarray) -> np.ndarray:
        """
        Compute diagonal of the kernel matrix (all ones for Tanimoto kernel).

        Args:
            X: Input array.

        Returns:
            Array of ones with shape (n_samples,).
        """
        return np.ones(X.shape[0])

    def is_stationary(self) -> bool:
        """
        Return whether the kernel is stationary (False for Tanimoto).

        Returns:
            False (Tanimoto kernel is not stationary).
        """
        return False


class PXRRegressor:
    """
    PXR Regressor using Gaussian Process Regression with Tanimoto kernel.
    
    Predicts PXR activation EC50 values from SMILES strings.

    Example:
        >>> regressor = PXRRegressor()
        >>> regressor.fit(smiles_list, ec50_values)
        >>> predictions, uncertainties = regressor.predict(test_smiles)
    """
    
    def __init__(self, model_file: Optional[str] = None):
        """
        Initialize the PXR regressor.

        Args:
            model_file: Optional path to a saved model file to load.

        Raises:
            FileNotFoundError: If model_file is provided but doesn't exist.
        """
        self.smiles_to_fp_converter = uru.Smi2Fp()
        if model_file is not None:
            try:
                with open(model_file, 'rb') as f:
                    state = pickle.load(f)
                    self.gpr = state['gpr']
            except FileNotFoundError:
                raise FileNotFoundError(f"Model file not found: {model_file}")
        else:
            kernel = TanimotoKernel()
            self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True)

    def fit(self, smiles_list: List[str], y_list: List[float]) -> 'PXRRegressor':
        """
        Fit the GPR model on SMILES strings and target values.

        Args:
            smiles_list: List of SMILES strings.
            y_list: List of target EC50 values.

        Returns:
            self: Returns self for method chaining.
        """
        X_fp = [self.smiles_to_fp_converter.get_np(x) for x in smiles_list]
        # Filter out None fingerprints
        valid_indices = [i for i, fp in enumerate(X_fp) if fp is not None]
        if not valid_indices:
            raise ValueError("No valid fingerprints could be computed from SMILES input.")
        X_fp_valid = [X_fp[i] for i in valid_indices]
        y_valid = [y_list[i] for i in valid_indices]
        self.gpr.fit(X_fp_valid, y_valid)
        return self

    def predict(self, smiles_list: List[str]) -> Tuple[List[float], List[float]]:
        """
        Predict EC50 values and uncertainties for input SMILES.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Tuple of (predictions, uncertainties) where both are lists of floats.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not hasattr(self.gpr, 'X_train_') or self.gpr.X_train_ is None:
            raise RuntimeError("Model must be fitted before calling predict. Call fit() first.")
        X_fp = [self.smiles_to_fp_converter.get_np(x) for x in smiles_list]
        # Filter out None fingerprints and track valid indices
        valid_indices = [i for i, fp in enumerate(X_fp) if fp is not None]
        if not valid_indices:
            raise ValueError("No valid fingerprints could be computed from SMILES input.")
        X_fp_valid = [X_fp[i] for i in valid_indices]
        y_pred, y_std = self.gpr.predict(X_fp_valid, return_std=True)
        # Create full-length lists with NaN for invalid SMILES
        predictions = [float('nan')] * len(smiles_list)
        uncertainties = [float('nan')] * len(smiles_list)
        for idx, valid_idx in enumerate(valid_indices):
            predictions[valid_idx] = float(y_pred[idx])
            uncertainties[valid_idx] = float(y_std[idx])
        return predictions, uncertainties

    def save(self, model_file: str) -> None:
        """
        Save the trained GPR model to a file.

        Args:
            model_file: Path to save the model file.
        """
        state = {'gpr': self.gpr}
        with open(model_file, 'wb') as f:
            pickle.dump(state, f)
