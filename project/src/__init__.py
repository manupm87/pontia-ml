"""Paquete `src`: sistema automático de entrenamiento y comparación de modelos
de clasificación binaria para la predicción de cancelaciones de reservas
hoteleras.

Módulos principales:

- ``config``        : configuración y constantes del proyecto.
- ``data_loader``   : carga, limpieza y partición de los datos.
- ``preprocessing`` : construcción del preprocesador (ColumnTransformer).
- ``model_trainer`` : definición y entrenamiento de los modelos.
- ``evaluator``     : cálculo de métricas y visualizaciones.
- ``train``         : orquestador del flujo completo (script principal).
- ``predict``       : inferencia con el mejor modelo seleccionado.

¿No conoces algún término técnico (one-hot, ROC-AUC, Pipeline, dropout...)? Todos
están explicados en lenguaje sencillo en ``docs/glosario.md``.
"""

__version__ = "1.0.0"
