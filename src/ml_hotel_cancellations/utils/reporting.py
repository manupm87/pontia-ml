"""Utilidades de reporte compartidas: tabla Markdown y guardado de figuras."""

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

    Con ``index_label=None`` el índice no se incluye como columna.
    """
    cols = list(df.columns)

    def fmt(value) -> str:
        if isinstance(value, float):
            return float_fmt.format(value)
        return str(value)

    if index_label is not None:
        header = f"| {index_label} | " + " | ".join(str(c) for c in cols) + " |"
        separator = "|" + "---|" * (len(cols) + 1)
        rows = [
            "| " + str(idx) + " | " + " | ".join(fmt(v) for v in row) + " |"
            for idx, row in zip(df.index, df.to_numpy())
        ]
    else:
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        separator = "|" + "---|" * len(cols)
        rows = [
            "| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.to_numpy()
        ]
    return "\n".join([header, separator, *rows])


def save_figure(fig, path, *, dpi: int = 120, bbox_inches: str | None = None) -> None:
    """Aplica ``tight_layout``, guarda la figura, la cierra y lo registra en el log."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    save_kwargs = {"dpi": dpi}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    logger.info("Gráfico guardado: %s", path)
