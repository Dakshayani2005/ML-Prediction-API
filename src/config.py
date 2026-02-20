from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    IMAGE_SIZE: tuple = (160, 160)
    MODEL_PATH: str = "models/my_classifier_model.h5"
    CLASS_NAMES: list = ["cats", "dogs"]
    LOG_LEVEL: str = "INFO"
    THRESHOLD: float = 0.85 

settings = Settings()