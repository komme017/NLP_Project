"""Shared Azure OpenAI client, built once from environment variables."""

import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

_client = None


def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    return _client


def get_deployment() -> str:
    return os.environ["AZURE_OPENAI_DEPLOYMENT"]
