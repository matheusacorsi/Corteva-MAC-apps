"""Optional deep-learning upgrade path (1D-CNN over raw windows instead of
hand-engineered features). Behind the ``deep`` extra (``pip install
epg-tool[deep]``) -- not required for the tabular baseline and not
implemented yet; this stub keeps the import path stable for when it is.
"""

from __future__ import annotations


def cnn1d_model(*args, **kwargs):
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 1D-CNN upgrade path requires torch. Install it with "
            "`pip install epg-tool[deep]` (or `pip install torch`)."
        ) from exc
    raise NotImplementedError(
        "1D-CNN sequence model is not implemented yet. The tabular "
        "Random Forest / XGBoost models (epg_tool.models.tabular) are the "
        "current baseline; this is the planned upgrade path once more "
        "labeled recordings are available."
    )
