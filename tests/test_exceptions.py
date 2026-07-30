# -*- coding: utf-8 -*-
"""Tests for the scomp_link.exceptions hierarchy."""

import pytest

from scomp_link.exceptions import (
    ArtifactError,
    ArtifactVersionError,
    ConfigError,
    DataLoadError,
    DataValidationError,
    DriftDetectionError,
    ModelTrainingError,
    PermissionUpdateError,
    ScompLinkError,
    UpdateError,
)


class TestExceptionHierarchy:
    """Verify that each exception is a subclass of the correct parent."""

    def test_data_load_error_is_data_validation_error(self):
        assert issubclass(DataLoadError, DataValidationError)

    def test_data_load_error_is_scomp_link_error(self):
        assert issubclass(DataLoadError, ScompLinkError)

    def test_data_validation_error_is_scomp_link_error(self):
        assert issubclass(DataValidationError, ScompLinkError)

    def test_model_training_error_is_scomp_link_error(self):
        assert issubclass(ModelTrainingError, ScompLinkError)

    def test_artifact_version_error_is_artifact_error(self):
        assert issubclass(ArtifactVersionError, ArtifactError)

    def test_artifact_error_is_scomp_link_error(self):
        assert issubclass(ArtifactError, ScompLinkError)

    def test_drift_detection_error_is_scomp_link_error(self):
        assert issubclass(DriftDetectionError, ScompLinkError)

    def test_permission_update_error_is_update_error(self):
        assert issubclass(PermissionUpdateError, UpdateError)

    def test_update_error_is_scomp_link_error(self):
        assert issubclass(UpdateError, ScompLinkError)

    def test_config_error_is_scomp_link_error(self):
        assert issubclass(ConfigError, ScompLinkError)


class TestExceptionInstances:
    """Verify isinstance checks work on raised instances."""

    def test_data_load_error_instance_caught_as_data_validation(self):
        with pytest.raises(DataValidationError):
            raise DataLoadError("file not found")

    def test_data_load_error_instance_caught_as_scomp_link(self):
        with pytest.raises(ScompLinkError):
            raise DataLoadError("file not found")

    def test_artifact_version_error_instance_caught_as_artifact_error(self):
        with pytest.raises(ArtifactError):
            raise ArtifactVersionError("version mismatch")

    def test_permission_update_error_instance_caught_as_update_error(self):
        with pytest.raises(UpdateError):
            raise PermissionUpdateError("read-only")

    def test_permission_update_error_instance_caught_as_scomp_link(self):
        with pytest.raises(ScompLinkError):
            raise PermissionUpdateError("read-only")

    def test_all_errors_caught_by_base(self):
        errors = [
            DataLoadError("x"),
            DataValidationError("x"),
            ModelTrainingError("x"),
            ArtifactError("x"),
            ArtifactVersionError("x"),
            DriftDetectionError("x"),
            UpdateError("x"),
            PermissionUpdateError("x"),
            ConfigError("x"),
        ]
        for err in errors:
            assert isinstance(err, ScompLinkError), f"{type(err).__name__} is not a ScompLinkError"

    def test_exception_message_preserved(self):
        msg = "test error message"
        err = DataLoadError(msg)
        assert str(err) == msg

    def test_exceptions_are_exceptions(self):
        for cls in (
            ScompLinkError,
            DataValidationError,
            DataLoadError,
            ModelTrainingError,
            ArtifactError,
            ArtifactVersionError,
            DriftDetectionError,
            UpdateError,
            PermissionUpdateError,
            ConfigError,
        ):
            assert issubclass(cls, Exception)
