"""Detección y uso opcional de GPU (CUDA).

El proyecto se ejecuta en **CPU por defecto**: para este tamaño de datos la GPU
no acelera (el coste por modelo es pequeño) y la CPU asegura resultados
reproducibles. Aun así el código es **"GPU-aware"**: activando
``PONTIA_USE_GPU=1`` (ver ``config.USE_GPU``) se usa la GPU en XGBoost
(``device='cuda'``) cuando haya una disponible, con caída automática a CPU si
no funciona.

Los modelos de scikit-learn (regresión logística, árbol, Random Forest) son
solo-CPU por diseño. La red neuronal usaría GPU **si** TensorFlow estuviese
instalado con soporte CUDA; con el paquete ``tensorflow-cpu`` se entrena en CPU.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from . import config

logger = logging.getLogger(__name__)

_xgb_device_cache: str | None = None


def cuda_available() -> bool:
    """Devuelve True si ``nvidia-smi`` detecta al menos una GPU NVIDIA."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5)
        return out.returncode == 0 and b"GPU" in out.stdout
    except Exception:
        return False


def xgboost_device() -> str:
    """Devuelve ``'cuda'`` si XGBoost puede entrenar en GPU; si no, ``'cpu'``.

    Hace una pequeña prueba real de entrenamiento en GPU (cacheada): así solo se
    usa CUDA si de verdad funciona en esta máquina (por ejemplo, que la
    arquitectura de la GPU esté soportada por la versión de XGBoost), evitando
    fallos en equipos sin GPU compatible.
    """
    global _xgb_device_cache
    if _xgb_device_cache is not None:
        return _xgb_device_cache

    device = "cpu"
    if config.USE_GPU and cuda_available():
        try:
            import numpy as np
            from xgboost import XGBClassifier

            X = np.random.rand(64, 4)
            y = (X[:, 0] > 0.5).astype(int)
            XGBClassifier(n_estimators=2, tree_method="hist", device="cuda").fit(X, y)
            device = "cuda"
        except Exception as exc:  # GPU presente pero XGBoost no puede usarla
            logger.warning(
                "GPU detectada pero XGBoost no puede usarla (%s); se usará CPU.",
                type(exc).__name__,
            )
            device = "cpu"
    _xgb_device_cache = device
    return device


def xgboost_gpu_kwargs() -> dict:
    """kwargs de XGBoost según el dispositivo.

    En CPU devuelve ``{}`` (los valores por defecto de XGBoost), de modo que el
    comportamiento es idéntico al de no tener nada de esto y los resultados son
    reproducibles. En GPU activa ``tree_method='hist'`` + ``device='cuda'``.
    """
    if xgboost_device() == "cuda":
        return {"tree_method": "hist", "device": "cuda"}
    return {}


def log_status() -> None:
    """Registra en el log qué aceleración se va a usar."""
    device = xgboost_device()
    if device == "cuda":
        logger.info("Aceleración por hardware -> XGBoost en GPU (CUDA).")
    elif config.USE_GPU:
        logger.info("PONTIA_USE_GPU activo pero sin GPU utilizable -> CPU.")
    elif cuda_available():
        logger.info("GPU detectada pero no usada (CPU por defecto; PONTIA_USE_GPU=1 para activarla).")
    else:
        logger.info("Ejecutando en CPU.")
