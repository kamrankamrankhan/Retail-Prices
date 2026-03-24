"""
Model Registry Module.

Provides a centralized registry for managing and versioning
machine learning models.
"""

import os
import json
import joblib
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized model registry for managing ML models.

    Features:
    - Model versioning
    - Model metadata tracking
    - Model loading and saving
    - Performance tracking
    """

    def __init__(self, registry_dir: str = 'models/registry'):
        """
        Initialize ModelRegistry.

        Args:
            registry_dir: Directory for storing model files
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / 'registry.json'
        self.registry: Dict[str, Dict] = {}

        self._load_registry()

    def _load_registry(self):
        """Load registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded registry with {len(self.registry)} models")
            except Exception as e:
                logger.error(f"Error loading registry: {str(e)}")
                self.registry = {}

    def _save_registry(self):
        """Save registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
            logger.debug("Registry saved")
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")

    def register_model(self, model_name: str, model: Any,
                       model_type: str, metrics: Dict,
                       features: List[str], description: str = '') -> str:
        """
        Register a new model.

        Args:
            model_name: Name for the model
            model: Trained model object
            model_type: Type of model (e.g., 'random_forest')
            metrics: Performance metrics dictionary
            features: List of feature names used
            description: Model description

        Returns:
            Model version identifier
        """
        # Generate version
        existing_versions = [
            v for v in self.registry.keys()
            if v.startswith(model_name)
        ]
        version = len(existing_versions) + 1
        version_id = f"{model_name}_v{version}"

        # Save model file
        model_path = self.registry_dir / f"{version_id}.joblib"
        joblib.dump(model, model_path)

        # Create metadata
        metadata = {
            'name': model_name,
            'version': version,
            'version_id': version_id,
            'model_type': model_type,
            'description': description,
            'features': features,
            'metrics': metrics,
            'model_path': str(model_path),
            'created_at': datetime.now().isoformat(),
            'is_active': True
        }

        # Register
        self.registry[version_id] = metadata
        self._save_registry()

        logger.info(f"Registered model: {version_id}")
        return version_id

    def get_model(self, version_id: str) -> Optional[Any]:
        """
        Load a model by version ID.

        Args:
            version_id: Model version identifier

        Returns:
            Loaded model or None
        """
        if version_id not in self.registry:
            logger.error(f"Model not found: {version_id}")
            return None

        metadata = self.registry[version_id]
        model_path = Path(metadata['model_path'])

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model: {version_id}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def get_latest_model(self, model_name: str) -> Optional[Any]:
        """
        Get the latest version of a model.

        Args:
            model_name: Model name

        Returns:
            Latest model or None
        """
        versions = [
            v for v in self.registry.keys()
            if self.registry[v]['name'] == model_name
        ]

        if not versions:
            logger.warning(f"No versions found for model: {model_name}")
            return None

        latest_version = sorted(versions)[-1]
        return self.get_model(latest_version)

    def get_active_model(self, model_name: str) -> Optional[Any]:
        """
        Get the active version of a model.

        Args:
            model_name: Model name

        Returns:
            Active model or None
        """
        for version_id, metadata in self.registry.items():
            if metadata['name'] == model_name and metadata.get('is_active', False):
                return self.get_model(version_id)

        return self.get_latest_model(model_name)

    def set_active_model(self, version_id: str) -> bool:
        """
        Set a model version as active.

        Args:
            version_id: Model version identifier

        Returns:
            True if successful
        """
        if version_id not in self.registry:
            logger.error(f"Model not found: {version_id}")
            return False

        model_name = self.registry[version_id]['name']

        # Deactivate other versions
        for vid, metadata in self.registry.items():
            if metadata['name'] == model_name:
                metadata['is_active'] = False

        # Activate this version
        self.registry[version_id]['is_active'] = True
        self._save_registry()

        logger.info(f"Set active model: {version_id}")
        return True

    def list_models(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        List registered models.

        Args:
            model_name: Optional filter by model name

        Returns:
            List of model metadata
        """
        models = []
        for version_id, metadata in self.registry.items():
            if model_name is None or metadata['name'] == model_name:
                models.append(metadata)

        return sorted(models, key=lambda x: x['created_at'], reverse=True)

    def get_model_metrics(self, version_id: str) -> Optional[Dict]:
        """
        Get metrics for a model version.

        Args:
            version_id: Model version identifier

        Returns:
            Metrics dictionary or None
        """
        if version_id not in self.registry:
            return None

        return self.registry[version_id].get('metrics', {})

    def compare_models(self, version_ids: List[str]) -> Dict:
        """
        Compare metrics across model versions.

        Args:
            version_ids: List of version IDs to compare

        Returns:
            Comparison dictionary
        """
        comparison = {}

        for version_id in version_ids:
            if version_id in self.registry:
                comparison[version_id] = self.registry[version_id]['metrics']

        return comparison

    def delete_model(self, version_id: str) -> bool:
        """
        Delete a model version.

        Args:
            version_id: Model version identifier

        Returns:
            True if successful
        """
        if version_id not in self.registry:
            logger.error(f"Model not found: {version_id}")
            return False

        # Delete model file
        model_path = Path(self.registry[version_id]['model_path'])
        if model_path.exists():
            model_path.unlink()

        # Remove from registry
        del self.registry[version_id]
        self._save_registry()

        logger.info(f"Deleted model: {version_id}")
        return True

    def get_registry_summary(self) -> Dict:
        """
        Get summary of the registry.

        Returns:
            Summary dictionary
        """
        models = self.list_models()

        return {
            'total_models': len(models),
            'unique_model_names': len(set(m['name'] for m in models)),
            'model_types': list(set(m['model_type'] for m in models)),
            'latest_models': models[:5]
        }
