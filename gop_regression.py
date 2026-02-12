import os
from typing import List, Optional

import joblib
import numpy as np

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "finalized_linear_model.joblib",
)

_gop_regression_pipeline = joblib.load(_MODEL_PATH)


def predict_gop_regression_score(gop_scores: List[float]) -> Optional[float]:
    """Apply the trained regression pipeline (StandardScaler ->
    SequentialFeatureSelector -> LinearRegression) to per-sentence GOP
    scores for a speaker, replacing simple averaging.

    Returns None when the pipeline cannot be applied (e.g. the speaker
    doesn't have all 27 sentence scores yet), signaling the caller to
    fall back to simple averaging.
    """
    if not gop_scores:
        return None

    gop_array = np.array(gop_scores).reshape(1, -1)

    try:
        predicted_score = _gop_regression_pipeline.predict(gop_array)
        return float(predicted_score[0])
    except Exception:
        return None
