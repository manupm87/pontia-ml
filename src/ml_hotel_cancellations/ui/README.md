# Interfaz visual (Streamlit) — bonus

Aplicación web que sirve de **escaparate** del proyecto de predicción de
cancelaciones de reservas de hotel. Muestra los resultados de los modelos, sus
visualizaciones, una exploración de los datos (EDA) y un formulario que consume
la **API FastAPI** para predecir en vivo.

> Pensada con fines **didácticos**: cada pantalla explica los términos de ML que
> aparecen (ROC-AUC, matriz de confusión, SHAP, balanceo de clases...).

## Requisitos

- Python con las dependencias del proyecto instaladas (`pip install -e .`; ver
  `pyproject.toml`), que ya incluyen `streamlit`, `requests`, `plotly`, `pandas`
  y `matplotlib`.
- Para la sección de **Predicción**, la API FastAPI del proyecto en marcha.

## Cómo arrancar la interfaz

Desde la raíz del repo (importante: para que `import ui` resuelva bien):

```bash
streamlit run src/ml_hotel_cancellations/ui/app.py
```

Se abrirá en el navegador (por defecto `http://localhost:8501`). La navegación
entre secciones está en la **barra lateral**.

## Predicciones: necesitas la API

La sección "Predicción (API)" envía la reserva a la API y muestra la
probabilidad de cancelación. **La API debe estar arrancada**; si no, la interfaz
lo detecta y explica cómo hacerlo (no se rompe).

Arranca la API en otra terminal, desde la raíz del repo:

```bash
uvicorn ml_hotel_cancellations.api.main:app --host 0.0.0.0 --port 8000
```

### Variable de entorno `PONTIA_API_URL`

La interfaz busca la API en `http://localhost:8000` por defecto. Si corre en
otra URL (otro puerto, Docker, despliegue...), indícalo **antes** de lanzar
Streamlit:

```bash
export PONTIA_API_URL="http://mi-host:puerto"
streamlit run src/ml_hotel_cancellations/ui/app.py
```

## Secciones

1. **Resumen y resultados** — tabla comparativa de los 5 modelos, métricas
   destacadas del ganador (XGBoost, ROC-AUC 0.9564) y gráficos clave.
2. **Visualización de los modelos** — galería con todas las visualizaciones
   disponibles en `outputs/` (curvas ROC, matrices de confusión, importancia de
   variables, balanceo) con explicaciones.
3. **Predicción (API)** — formulario con las 27 variables de una reserva que
   consulta la API FastAPI y muestra la predicción.
4. **Interpretabilidad** — gráficos SHAP si existen (`outputs/shap_*.png`); si
   no, importancia de variables y aviso de que aparecerán al ejecutar el módulo.
5. **Exploración (EDA)** — tasa de cancelación por categoría, balance de clases
   y resumen de variables numéricas.

## Estructura del código (modular)

La lógica (carga de datos, agregaciones, llamadas a la API) está separada del
renderizado de cada página, para que sea fácil de probar e iterar.

```
ml_hotel_cancellations/ui/
├── app.py            # Punto de entrada: navegación y estado de la API.
├── layout.py         # Composición del layout y la barra lateral.
├── config.py         # Rutas, URL de la API (PONTIA_API_URL) y constantes.
├── data.py           # Helpers cacheados: métricas, dataset, EDA y cliente API.
├── booking.py        # Esquema de las 27 variables del formulario y su payload.
├── sections/         # Una función render() por sección (sin lógica de datos).
│   ├── resumen.py
│   ├── visualizaciones.py
│   ├── prediccion.py
│   ├── interpretabilidad.py
│   └── eda.py
└── README.md
```

## Prueba rápida (arranque sin navegador)

```bash
timeout 20 streamlit run src/ml_hotel_cancellations/ui/app.py --server.headless true --server.port 8599
```

Debería mostrar "You can now view your Streamlit app..." sin errores.
