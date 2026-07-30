# -*- coding: utf-8 -*-
"""
███████╗██╗  ██╗ ██████╗███████╗██████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗
██╔════╝╚██╗██╔╝██╔════╝██╔════╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
█████╗   ╚███╔╝ ██║     █████╗  ██████╔╝   ██║   ██║██║   ██║██╔██╗ ██║███████╗
██╔══╝   ██╔██╗ ██║     ██╔══╝  ██╔═══╝    ██║   ██║██║   ██║██║╚██╗██║╚════██║
███████╗██╔╝ ██╗╚██████╗███████╗██║        ██║   ██║╚██████╔╝██║ ╚████║███████║
╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝        ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝

Custom exception hierarchy for scomp-link.

All exceptions derive from ScompLinkError, allowing callers to handle either
a specific failure or the entire scomp-link error surface with a single clause:

    from scomp_link.exceptions import ScompLinkError, DataLoadError

    try:
        result = services.describe(config)
    except DataLoadError as e:
        print(f"Bad file: {e}")
    except ScompLinkError as e:
        print(f"scomp-link error: {e}")
"""


class ScompLinkError(Exception):
    """Base class for all scomp-link exceptions.

    Catch this to handle any error raised by the scomp-link package without
    importing the full exception hierarchy.
    """


# ── Data & Input ─────────────────────────────────────────────────────────────


class DataValidationError(ScompLinkError):
    """Raised when input data or arguments fail validation before processing.

    Examples:
        - Target column not found in DataFrame
        - DataFrame is empty or has fewer rows than required
        - Invalid parameter value (negative n_clusters, unknown task type)
        - Incompatible reference/current shapes for drift detection
    """


class DataLoadError(DataValidationError):
    """Raised when a file cannot be read or parsed.

    Subclass of DataValidationError — callers catching DataValidationError
    will also catch this.

    Examples:
        - File path does not exist
        - Unsupported file extension (.xlsx, .json, .xml)
        - CSV/Parquet file is corrupt or empty
        - Insufficient read permissions on the file
    """


# ── Model Training ────────────────────────────────────────────────────────────


class ModelTrainingError(ScompLinkError):
    """Raised when model training or hyperparameter tuning fails.

    Examples:
        - Fewer training samples than required by the selected algorithm
        - Incompatible feature types (e.g. strings passed to a numeric-only model)
        - All Optuna trials failed — no successful model found
        - Convergence failure in iterative solvers (LinearSVC, SGD)
    """


# ── Artifacts (.scomp) ───────────────────────────────────────────────────────


class ArtifactError(ScompLinkError):
    """Raised for any failure during .scomp artifact save or load.

    Examples:
        - Artifact file not found at the given path
        - File exists but is not a valid .scomp archive (bad magic bytes)
        - Pickle deserialization fails because a class no longer exists
        - ZIP archive is corrupt or truncated
    """


class ArtifactVersionError(ArtifactError):
    """Raised when an artifact was created with an incompatible format version.

    Subclass of ArtifactError. Callers can catch this specifically to emit
    migration advice, or catch ArtifactError to handle all artifact failures.

    Examples:
        - Artifact created with format version "1.0", current is "2.0"
        - Required manifest keys missing because of a schema change
    """


# ── Monitoring & Drift ────────────────────────────────────────────────────────


class DriftDetectionError(ScompLinkError):
    """Raised when drift detection cannot be performed.

    Examples:
        - Reference and current datasets share no common numeric columns
        - Fewer than 2 samples in reference or current dataset
        - PSI computation fails because all values fall into a single bin
    """


# ── MCP & Auto-update ─────────────────────────────────────────────────────────


class UpdateError(ScompLinkError):
    """Raised when the auto-update or manual update system fails.

    Examples:
        - No network connectivity to PyPI
        - pip install returns a non-zero exit code
        - Update timeout exceeded (default 120 s)
        - Unable to determine the installed package version
    """


class PermissionUpdateError(UpdateError):
    """Raised when pip install fails because site-packages is read-only.

    Subclass of UpdateError. Provides callers a typed signal to emit a
    friendlier message ("run pip install manually") without parsing error
    strings.

    Examples:
        - Running inside a Docker container with a read-only venv
        - Shared HPC environment where users cannot write to site-packages
        - System Python with no sudo access
    """


# ── Configuration ─────────────────────────────────────────────────────────────


class ConfigError(ScompLinkError):
    """Raised when a scomp-link configuration file is malformed or incomplete.

    Examples:
        - ~/.scomp-link/config.yaml contains invalid YAML syntax
        - .scomp-link.yaml is missing a required field (e.g. main_color)
        - Config value has wrong type (string where hex color is expected)
    """
