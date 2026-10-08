import json
import re
from typing import List
import requests


class OllamaClient:
    """Client for Ollama local AI API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def _escape_json_string(self, s: str) -> str:
        """Escape a string for JSON."""
        return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    def _parse_embedding(self, body: str) -> List[float]:
        """Parse embedding array from Ollama response."""
        try:
            data = json.loads(body)
            if 'embedding' in data:
                return data['embedding']
        except json.JSONDecodeError:
            pass

        # Fallback: manually parse array
        match = re.search(r'"embedding"\s*:\s*\[([^\]]+)\]', body)
        if match:
            try:
                return [float(x.strip()) for x in match.group(1).split(',')]
            except ValueError:
                pass
        return []

    def _parse_response(self, body: str) -> str:
        """Parse response field from Ollama generate response."""
        try:
            data = json.loads(body)
            if 'response' in data:
                return data['response']
        except json.JSONDecodeError:
            pass
        return ""

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text using Ollama.
        Returns empty list if Ollama is unavailable.
        """
        try:
            body = {
                "model": self.embed_model,
                "prompt": text
            }
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            if response.status_code == 200:
                return self._parse_embedding(response.text)
        except requests.RequestException:
            pass
        return []

    def generate(self, prompt: str) -> str:
        """
        Generate text using Ollama LLM.
        Returns error message if Ollama is unavailable.
        """
        try:
            body = {
                "model": self.gen_model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=180
            )
            if response.status_code == 200:
                return self._parse_response(response.text)
            return "ERROR: Ollama unavailable. Run: ollama serve"
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"
