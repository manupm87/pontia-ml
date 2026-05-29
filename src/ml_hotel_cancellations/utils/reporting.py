"""Utilidades de reporte compartidas (Markdown y guardado de figuras).

Concentra dos piezas de *boilerplate* que se repetían en varios módulos:

- :func:`df_to_markdown`: convierte un ``DataFrame`` a una tabla Markdown sin
  depender de ``tabulate``. Fuente única reutilizada por train/tuning/balancing.
- :func:`save_figure`: encapsula el patrón ``tight_layout(); savefig(); close();
  logger.info(...)`` que aparecía en evaluator/interpretability/balancing/
  visualization_2d, estandarizando el DPI.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def df_to_markdown(
    df,
    *,
    index_label: str | None = "Modelo",
    float_fmt: str = "{:.4f}",
) -> str:
    """Convierte un ``DataFrame`` a una tabla Markdown (sin dependencias externas).

    Parameters
    ----------
    df:
        DataFrame a renderizar.
    index_label:
        Encabezado de la columna del índice. Si es ``None``, el índice NO se
        incluye como columna (se renderizan solo las columnas de datos).
    float_fmt:
        Formato aplicado a los valores ``float`` (p. ej. ``"{:.4f}"``). Los
        valores no flotantes se convierten con ``str``.
    """
    cols = list(df.columns)

    def fmt(value) -> str:
        if isinstance(value, float):
            return float_fmt.format(value)
        return str(value)

    if index_label is not None:
        encabezado = f"| {index_label} | " + " | ".join(str(c) for c in cols) + " |"
        separador = "|" + "---|" * (len(cols) + 1)
        filas = [
            "| " + str(idx) + " | " + " | ".join(fmt(v) for v in fila) + " |"
            for idx, fila in zip(df.index, df.to_numpy())
        ]
    else:
        encabezado = "| " + " | ".join(str(c) for c in cols) + " |"
        separador = "|" + "---|" * len(cols)
        filas = [
            "| " + " | ".join(fmt(v) for v in fila) + " |"
            for fila in df.to_numpy()
        ]
    return "\n".join([encabezado, separador, *filas])


def save_figure(fig, path, *, dpi: int = 120, bbox_inches: str | None = None) -> None:
    """Aplica ``tight_layout``, guarda la figura, la cierra y lo registra en el log.

    Estandariza el guardado de figuras (DPI=120 por defecto) que antes se repetía
    con valores dispares (110/120/130) por todo el código.

    Parameters
    ----------
    fig:
        Figura de matplotlib a guardar.
    path:
        Ruta del PNG de salida.
    dpi:
        Resolución del PNG.
    bbox_inches:
        Si se indica (``"tight"``), se pasa a ``savefig`` para recortar márgenes.
    """
    import matplotlib.pyplot as plt

    fig.tight_layout()
    save_kwargs = {"dpi": dpi}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    logger.info("Gráfico guardado: %s", path)
