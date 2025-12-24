from dataclasses import dataclass

@dataclass
class BaseArtifact:
    artifact_dir: str

class DataIngestionArtifact(BaseArtifact):
    train_file_path: str
    test_file_path: str
