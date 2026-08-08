"""Classification tool handler."""

from __future__ import annotations

import json
from typing import Any

from eo_processor_mcp.utils import format_result, load_array, save_array


def handle_classify(args: dict[str, Any]) -> str:
    """Train or predict with Random Forest, or run complex classification."""
    import eo_processor as eop

    method = args["method"].lower()

    if method == "train":
        features = load_array(args["features"])
        labels = load_array(args["labels"])
        n_estimators = int(args.get("n_estimators", 100))
        min_samples_split = int(args.get("min_samples_split", 2))
        max_depth = args.get("max_depth")
        if max_depth is not None:
            max_depth = int(max_depth)
        max_features = args.get("max_features")
        if max_features is not None:
            max_features = int(max_features)

        model_json = eop.random_forest_train(
            features,
            labels,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            max_features=max_features,
        )
        return json.dumps(
            {
                "method": "train",
                "model": model_json,
                "n_estimators": n_estimators,
                "min_samples_split": min_samples_split,
                "max_depth": max_depth,
                "max_features": max_features,
            },
            separators=(",", ":"),
        )

    if method == "predict":
        features = load_array(args["features"])
        model_json = args["model"]
        output_path = args.get("output_path")

        result = eop.random_forest_predict(model_json, features)
        out = save_array(result, output_path)
        return format_result(out, result, {"method": "predict"})

    if method == "complex":
        blue = load_array(args["blue"])
        green = load_array(args["green"])
        red = load_array(args["red"])
        nir = load_array(args["nir"])
        swir1 = load_array(args["swir1"])
        swir2 = load_array(args["swir2"])
        temp = load_array(args["temp"])
        output_path = args.get("output_path")

        result = eop.complex_classification(blue, green, red, nir, swir1, swir2, temp)
        out = save_array(result, output_path)
        return format_result(out, result, {"method": "complex_classification"})

    msg = f"Unknown classify method: {method}. Available: [train, predict, complex]"
    raise ValueError(msg)
