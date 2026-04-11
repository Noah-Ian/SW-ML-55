"""Feature layout matches agro_moisture.ipynb (RobustScaler input order)."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

FEATURE_NAMES: list[str] = [
    "Soil_pH",
    "Temperature(C)",
    "Humidity(%)",
    "Fertilizer_Recommended(kg/ha)",
    "Irrigation_Recommended(mm)",
    "Month_sin",
    "Month_cos",
    "Moisture_Lag1",
    "Moisture_Lag2",
    "Moisture_RollingMean3",
    "Crop_Type_Lettuce",
    "Crop_Type_Maize",
    "Crop_Type_Tomatoes",
    "Crop_Type_Wheat",
]

CropType = Literal["Beans", "Lettuce", "Maize", "Tomatoes", "Wheat"]


def month_cyclical(month: int) -> tuple[float, float]:
    if not 1 <= month <= 12:
        raise ValueError("month must be 1–12")
    rad = 2 * math.pi * month / 12
    return math.sin(rad), math.cos(rad)


def crop_one_hot(crop: CropType) -> tuple[float, float, float, float]:
    """pd.get_dummies(..., drop_first=True) drops Beans."""
    return (
        1.0 if crop == "Lettuce" else 0.0,
        1.0 if crop == "Maize" else 0.0,
        1.0 if crop == "Tomatoes" else 0.0,
        1.0 if crop == "Wheat" else 0.0,
    )


def build_feature_row(
    *,
    soil_ph: float,
    temperature_c: float,
    humidity_pct: float,
    fertilizer_kg_ha: float,
    irrigation_mm: float,
    month: int,
    crop: CropType,
    moisture_lag1: float,
    moisture_lag2: float,
    moisture_rolling_mean3: float,
) -> np.ndarray:
    ms, mc = month_cyclical(month)
    cl, cm, ct, cw = crop_one_hot(crop)
    return np.array(
        [
            soil_ph,
            temperature_c,
            humidity_pct,
            fertilizer_kg_ha,
            irrigation_mm,
            ms,
            mc,
            moisture_lag1,
            moisture_lag2,
            moisture_rolling_mean3,
            cl,
            cm,
            ct,
            cw,
        ],
        dtype=np.float32,
    ).reshape(1, -1)
