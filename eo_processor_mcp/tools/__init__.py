"""Tool handler constants."""

SPECTRAL_INDICES_2BAND = [
    "ndvi", "ndwi", "ndsi", "evi2", "savi", "osavi", "msavi",
    "gndvi", "ndre", "nbr", "ndmi", "nbr2", "gci", "ci_re",
]

SPECTRAL_INDICES_3BAND = [
    "evi", "lai", "ndvi_re2", "mtci",
]

CHANGE_INDICES = [
    "delta_ndvi", "delta_nbr", "dnbr", "rbr",
]

TEMPORAL_METHODS = ["median", "mean", "std", "sum"]

MASK_METHODS = [
    "vals", "replace_nans", "out_range", "in_range",
    "invalid", "scl", "with_scl",
]

MORPHOLOGY_OPS = ["dilation", "erosion", "opening", "closing"]

DISTANCE_METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski"]

ALL_TOOLS = [
    "compute_spectral_index",
    "compute_change_index",
    "temporal_statistics",
    "temporal_composite",
    "moving_average",
    "apply_mask",
    "morphological_operation",
    "compute_distances",
    "analyze_trends",
    "bfast_monitor",
    "classify",
    "texture_features",
    "zonal_statistics",
    "pixelwise_transform",
    "list_capabilities",
]
