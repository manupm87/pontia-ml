"""API REST (bonus) para servir el modelo de cancelaciones de reservas.

Este paquete expone el mejor modelo entrenado (un ``Pipeline`` de scikit-learn
con XGBoost) a través de una API HTTP construida con FastAPI. La idea es separar
responsabilidades de forma clara y didáctica:

- ``schemas``  : contratos de entrada/salida (modelos Pydantic).
- ``service``  : carga del modelo (una sola vez) y lógica de predicción,
                 reutilizando el preprocesado de ``src`` para ser idénticos al
                 entrenamiento.
- ``main``     : la aplicación FastAPI con los endpoints HTTP.

De esta forma una interfaz (p. ej. Streamlit) puede consumir las predicciones
sin conocer los detalles internos del modelo.
"""
