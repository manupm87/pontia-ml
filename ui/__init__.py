"""Paquete de la interfaz visual (Streamlit) del proyecto.

Agrupa la aplicación web que sirve de "escaparate" del proyecto de ML:
muestra los resultados de los modelos, sus visualizaciones, una exploración
de los datos (EDA) y un formulario que consume la API de predicción.

El código está separado en módulos para que la lógica (carga de datos,
agregaciones, llamadas a la API) sea fácil de probar e iterar, y quede
desacoplada del renderizado de cada página de Streamlit.
"""
