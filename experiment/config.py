from pathlib import Path
from pydantic import BaseModel

class ConfigTrain(BaseModel):
    train_size: int
    validation_size: int
    batch_size: int
    num_workers: int
    radius: float
    path: str
    split_file: str

class ConfigModel(BaseModel):
    vocab_dim: int
    feature_dim: int

class ConfigExperiment(BaseModel):
    model: ConfigModel
    train: ConfigTrain

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "ConfigExperiment":
        with open(path, "r") as f:
            json_config = f.read()
        return cls.model_validate_json(json_config)

    def save(self, path: str | Path):
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

class BuilderConfigExperiment:
    """
    A builder for `ConfigExperiment` that applies default values
    so the BaseModel classes don't have to define their own.
    """

    def __init__(self):
        # Centralize defaults here.
        self._model_config = {
            "vocab_dim": 10,
            "feature_dim": 128,
        }
        self._train_config = {
            "train_size": 110_000,
            "validation_size": 10_000,
            "batch_size": 32,
            "num_workers": 6,
            "radius": 3,
            "path": "",
            "split_file": "",
        }

    # ====== Model Config Methods ======
    def set_vocab_dim(self, vocab_dim: int):
        self._model_config["vocab_dim"] = vocab_dim
        return self

    def set_feature_dim(self, feature_dim: int):
        self._model_config["feature_dim"] = feature_dim
        return self

    # ====== Train Config Methods ======
    def set_train_size(self, train_size: int):
        self._train_config["train_size"] = train_size
        return self

    def set_validation_size(self, validation_size: int):
        self._train_config["validation_size"] = validation_size
        return self

    def set_batch_size(self, batch_size: int):
        self._train_config["batch_size"] = batch_size
        return self

    def set_num_workers(self, num_workers: int):
        self._train_config["num_workers"] = num_workers
        return self

    def set_radius(self, radius: float):
        self._train_config["radius"] = radius
        return self

    def set_path(self, path: str | Path):
        self._train_config["path"] = str(path)
        return self

    def set_split_file(self, split_file: str | Path):
        self._train_config["split_file"] = str(split_file)
        return self

    # ====== Example: Profile method ======
    def select_experiment_profile(self, profile: str):
        """
        Set multiple fields at once based on a 'profile'
        """
        if profile == "default":
            self._model_config["vocab_dim"] = 10
            self._model_config["feature_dim"] = 128
            self._train_config["batch_size"] = 32
            self._train_config["radius"] = 3
        elif profile == "small":
            self._model_config["vocab_dim"] = 8
            self._model_config["feature_dim"] = 64
            self._train_config["batch_size"] = 16
            self._train_config["radius"] = 2
        else:
            raise ValueError(f"Unknown experiment profile: {profile}")
        return self

    def build(self) -> ConfigExperiment:
        model = ConfigModel(**self._model_config)
        train = ConfigTrain(**self._train_config)
        return ConfigExperiment(model=model, train=train)

# Example usage:
if __name__ == "__main__":
    config = (
        BuilderConfigExperiment()
        .set_vocab_dim(20)
        .set_feature_dim(256)
        .set_train_size(150_000)
        .set_validation_size(15_000)
        .set_batch_size(64)
        .set_num_workers(12)
        .set_path("some/data/dataset/")
        .set_split_file("splits/train_val.json")
        .build()
    )

    print(config.model_dump_json(indent=4))
