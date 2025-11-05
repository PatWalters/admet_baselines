"""
CYP (Cytochrome P450) classifier models for ADMET prediction.

This module provides classes for predicting CYP enzyme inhibition and substrate properties
from SMILES strings using LightGBM classifiers.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import useful_rdkit_utils as uru
import lightgbm as lgb
import pickle
import warnings


class CYPModel:
    """
    CYPModel: A wrapper for LightGBM classifier to work directly with SMILES input.
    
    Handles fingerprint generation, missing/invalid SMILES, and provides convenience methods.
    This is a base class for individual CYP models.
    
    Example:
        >>> model = CYPModel()
        >>> model.fit(smiles_list, labels)
        >>> predictions = model.predict(test_smiles)
    """

    def __init__(
        self, 
        clf: Optional[lgb.LGBMClassifier] = None, 
        smi2fp: Optional[uru.Smi2Fp] = None, 
        verbose: bool = False
    ):
        """
        Initialize the CYP model.

        Args:
            clf: LightGBM classifier instance. If None, will use LightGBM's LGBMClassifier.
            smi2fp: Optional Smi2Fp instance for SMILES-to-fingerprint conversion.
            verbose: Whether to print warnings for invalid SMILES.
        """
        if clf is None:
            self.clf = lgb.LGBMClassifier(verbose=-1)
        else:
            self.clf = clf
        self.is_fitted = False
        self.smi2fp = smi2fp if smi2fp is not None else uru.Smi2Fp()
        self.verbose = verbose
        self.class_names_ = None

    def _smiles_to_fps(self, smiles: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Convert a list of SMILES to a 2D numpy array of fingerprints.
        
        Handles or warns about invalid/missing SMILES.

        Args:
            smiles: Iterable of SMILES strings.

        Returns:
            Tuple of:
                fps: np.ndarray of valid fingerprints (n_valid, n_features)
                mask: List of indices corresponding to valid fingerprints

        Raises:
            ValueError: If no valid fingerprints could be computed.
        """
        fps = []
        mask = []
        for i, smi in enumerate(smiles):
            fp = self.smi2fp.get_np(smi)
            if fp is not None:
                fps.append(fp)
                mask.append(i)
            else:
                if self.verbose:
                    warnings.warn(f"Invalid or unparsable SMILES at position {i}: {smi}")
        if not fps:
            raise ValueError("No valid fingerprints could be computed from SMILES input.")
        return np.stack(fps), mask

    def fit(self, smiles: List[str], y: List[Any]) -> 'CYPModel':
        """
        Fit the model on SMILES strings and labels.

        Args:
            smiles: List-like of SMILES strings (n_samples,)
            y: Class labels (n_samples,)

        Returns:
            self: Returns self for method chaining
        """
        fps, mask = self._smiles_to_fps(smiles)
        y = np.array(y)[mask]
        self.clf.fit(fps, y)
        self.is_fitted = True
        self.class_names_ = self.clf.classes_
        return self

    def predict(
        self, 
        smiles: List[str], 
        return_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
        """
        Predict the class labels for input SMILES.

        Args:
            smiles: List-like of SMILES strings
            return_mask: If True, additionally return mask of valid SMILES indices

        Returns:
            If return_mask=False: preds (predicted class labels for valid SMILES)
            If return_mask=True: Tuple of (preds, mask) where mask is list of valid indices

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict. Call fit() first.")
        fps, mask = self._smiles_to_fps(smiles)
        preds = self.clf.predict(fps)
        if return_mask:
            return preds, mask
        return preds

    def predict_proba(
        self, 
        smiles: List[str], 
        return_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
        """
        Return predicted class probabilities.

        Args:
            smiles: List-like of SMILES strings
            return_mask: If True, also return mask of valid SMILES indices

        Returns:
            If return_mask=False: probas (array of shape (n_valid, n_classes))
            If return_mask=True: Tuple of (probas, mask)

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba. Call fit() first.")
        fps, mask = self._smiles_to_fps(smiles)
        probas = self.clf.predict_proba(fps)
        if return_mask:
            return probas, mask
        return probas

    def save(self, filename: str) -> None:
        """
        Save the current CYPModel instance using pickle.

        Args:
            filename: Path to save the model file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> 'CYPModel':
        """
        Load a CYPModel instance from pickle.

        Args:
            filename: Path to the saved model file.

        Returns:
            Loaded CYPModel instance
        """
        with open(filename, "rb") as f:
            return pickle.load(f)


class CYPClassifier:
    """
    Wrapper for multiple CYPClassifier models, organized by key.
    
    Allows prediction and probability estimation for sets of SMILES strings
    across multiple CYP models (e.g., different isoforms or inhibition/substrate).

    Example:
        >>> classifier = CYPClassifier('CYP_classifier.pkl')
        >>> predictions = classifier.predict('1A2_Inh', smiles_list)
        >>> probabilities = classifier.predict_proba('2C9_Sub', smiles_list)
    """

    def __init__(
        self, 
        model_dict_or_filename: Union[Dict[str, Any], str], 
        smi2fp: Optional[uru.Smi2Fp] = None
    ):
        """
        Initialize the CYP classifier with a model dictionary or filename.

        Args:
            model_dict_or_filename: Dictionary of models, or filename to load models from.
            smi2fp: Optional Smi2Fp instance. If None, will be created as needed.

        Raises:
            ValueError: If input is neither a dictionary nor a filename string.
            FileNotFoundError: If filename doesn't exist.
        """
        if isinstance(model_dict_or_filename, dict):
            self.model_dict = model_dict_or_filename
        elif isinstance(model_dict_or_filename, str):
            try:
                with open(model_dict_or_filename, "rb") as f:
                    self.model_dict = pickle.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Model file not found: {model_dict_or_filename}")
        else:
            raise ValueError("Input must be a dictionary or a filename string.")
        # Initialize Smi2Fp interface only once if not passed
        self.smi2fp = smi2fp if smi2fp is not None else uru.Smi2Fp()

    def predict(
        self, 
        key: str, 
        smiles_list: List[str], 
        suppress_warnings: bool = True
    ) -> np.ndarray:
        """
        Predict class labels for input SMILES using a specific model.

        Args:
            key: Model key (e.g., '1A2_Inh', '2C9_Sub').
            smiles_list: List of SMILES strings.
            suppress_warnings: Whether to suppress warnings (default True).

        Returns:
            np.ndarray of predictions.

        Raises:
            KeyError: If model key is not found in the dictionary.
        """
        if key not in self.model_dict:
            available = ", ".join(self.model_dict.keys())
            raise KeyError(
                f"Model with key '{key}' not found in dictionary. "
                f"Available keys: {available}"
            )
        clf = self.model_dict[key]
        fp_list = [self.smi2fp.get_np(smi) for smi in smiles_list]
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                pred = clf.predict(fp_list)
        else:
            pred = clf.predict(fp_list)
        return pred

    def predict_proba(
        self, 
        key: str, 
        smiles_list: List[str], 
        suppress_warnings: bool = True, 
        return_classes: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate class probabilities for input SMILES using a specific model.

        Args:
            key: Model key (e.g., '1A2_Inh', '2C9_Sub').
            smiles_list: List of SMILES strings.
            suppress_warnings: Whether to suppress warnings (default True).
            return_classes: If True, also return ordered class labels.

        Returns:
            If return_classes=False: probs (np.ndarray of shape (n_samples, n_classes))
            If return_classes=True: Tuple of (probs, classes) where classes are class labels

        Raises:
            KeyError: If model key is not found in the dictionary.
        """
        if key not in self.model_dict:
            available = ", ".join(self.model_dict.keys())
            raise KeyError(
                f"Model with key '{key}' not found in dictionary. "
                f"Available keys: {available}"
            )
        clf = self.model_dict[key]
        fp_list = [self.smi2fp.get_np(smi) for smi in smiles_list]
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                probs = clf.predict_proba(fp_list)
        else:
            probs = clf.predict_proba(fp_list)
        if return_classes:
            return probs, clf.classes_
        return probs

    def available_model_ids(self) -> List[str]:
        """
        Return a list of the available model IDs (keys).

        Returns:
            List of model keys (strings).
        """
        return list(self.model_dict.keys())

    def save(self, filename: str) -> None:
        """
        Save the dictionary of models to a file.

        Args:
            filename: Path to save the model dictionary.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.model_dict, f)
