from .image import compare_image
from .ml import compare_model
from .myjson import compare_json, compare_json_normalized
from .table import compare_csv, compare_sqlite
from .text import compare_text

__all__ = [
    "compare_csv",
    "compare_image",
    "compare_json",
    "compare_json_normalized",
    "compare_model",
    "compare_sqlite",
    "compare_text",
]
