from dataclasses import dataclass

@dataclass
class BaseArtifact:
    artifact_dir: str

@dataclass
class DataIngestionArtifact(BaseArtifact):
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str
    preprocessing_object_path: str
