from chromadb.api.types import Embeddings, Documents, EmbeddingFunction
from typing import List, Dict, Any, Optional
import numpy as np
from tinfoil import TinfoilAI


class TinfoilAIEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        api_key: Optional[str] = None,
        enclave: str = "",
        repo: str = "",
        model_name: str = "",
    ):
        self.api_key = api_key
        self.repo = repo
        self.enclave = enclave
        self.model_name = model_name

        self.client = TinfoilAI(
            api_key=self.api_key,
            enclave=self.enclave,
            repo=self.repo,
        )

    def __call__(self, input: Documents) -> Embeddings:
        # Handle batching
        if not input:
            return []

        # Prepare embedding parameters
        embedding_params: Dict[str, Any] = {
            "model": self.model_name,
            "input": input,
        }

        # Get embeddings
        response = self.client.embeddings.create(**embedding_params)

        # Extract embeddings from response
        return [np.array(data.embedding, dtype=np.float32) for data in response.data]

    @staticmethod
    def name() -> str:
        return "tinfoil"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction[Documents]":
        # Extract parameters from config
        api_key = config.get("api_key")
        model_name = config.get("model_name")
        enclave = config.get("enclave", "")
        repo = config.get("repo", "")

        if api_key is None or model_name is None:
            raise ValueError("api_key and model_name are required parameters")

        # Create and return the embedding function
        return TinfoilAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            enclave=enclave,
            repo=repo,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "model_name": self.model_name,
            "enclave": self.enclave,
            "repo": self.repo,
        }

    def validate_config_update(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        if "model_name" in new_config and new_config["model_name"] != old_config["model_name"]:
            raise ValueError(
                "The model name cannot be changed after the embedding function has been initialized."
            )
        if "api_key" in new_config and new_config["api_key"] != old_config["api_key"]:
            raise ValueError(
                "The API key cannot be changed after the embedding function has been initialized."
            )

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        required_fields = ["api_key", "model_name"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
