"""Strict payload validation schemas for the serving API."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import HTTPException


def validate_batch_payload(payload: dict[str, Any], max_features: int, max_batch_rows: int) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """Validate and extract batch payload fields.
    
    Args:
        payload: request JSON body
        max_features: maximum allowed feature dimension
        max_batch_rows: maximum allowed batch size
        
    Returns:
        (features_array, labels_array_or_none, metadata_dict)
        
    Raises:
        HTTPException: on validation failure
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    
    # Validate features
    features_raw = payload.get("features")
    if features_raw is None:
        raise HTTPException(status_code=400, detail="features field is required")
    
    if not isinstance(features_raw, (list, tuple)):
        raise HTTPException(status_code=400, detail="features must be an array")
    
    if not features_raw:
        raise HTTPException(status_code=400, detail="features cannot be empty")
    
    try:
        features = np.asarray(features_raw, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"features must contain numeric values: {exc}") from exc
    
    if features.ndim == 1:
        features = features.reshape(1, -1)
    elif features.ndim != 2:
        raise HTTPException(status_code=400, detail=f"features must be 1D or 2D, got shape {features.shape}")
    
    if features.shape[0] > max_batch_rows:
        raise HTTPException(status_code=400, detail=f"batch exceeds max_batch_rows={max_batch_rows}, got {features.shape[0]}")
    
    if features.shape[1] > max_features:
        raise HTTPException(status_code=400, detail=f"feature dimension exceeds max={max_features}, got {features.shape[1]}")
    
    # Validate labels
    labels = None
    labels_raw = payload.get("labels")
    if labels_raw is not None:
        if not isinstance(labels_raw, (list, tuple)):
            raise HTTPException(status_code=400, detail="labels must be an array")
        try:
            labels = np.asarray(labels_raw, dtype=np.int64)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"labels must contain integer values: {exc}") from exc
        
        if labels.shape[0] != features.shape[0]:
            raise HTTPException(status_code=400, detail=f"labels shape {labels.shape[0]} does not match features {features.shape[0]}")
    
    # Validate metadata
    metadata = {}
    metadata_raw = payload.get("metadata")
    if metadata_raw is not None:
        if not isinstance(metadata_raw, dict):
            raise HTTPException(status_code=400, detail="metadata must be an object")
        metadata = dict(metadata_raw)
    
    return features, labels, metadata


def validate_label_payload(payload: dict[str, Any]) -> np.ndarray:
    """Validate and extract label payload.
    
    Args:
        payload: request JSON body
        
    Returns:
        labels_array
        
    Raises:
        HTTPException: on validation failure
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    
    labels_raw = payload.get("labels")
    if labels_raw is None:
        raise HTTPException(status_code=400, detail="labels field is required")
    
    if not isinstance(labels_raw, (list, tuple)):
        raise HTTPException(status_code=400, detail="labels must be an array")
    
    if not labels_raw:
        raise HTTPException(status_code=400, detail="labels cannot be empty")
    
    try:
        labels = np.asarray(labels_raw, dtype=np.int64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"labels must contain integer values: {exc}") from exc
    
    return labels
