from typing import Any, Optional
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler

from genai import Client, Credentials
from genai.extensions.langchain import LangChainInterface
from genai.text.generation import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationParameters,
)

load_dotenv()

watsonx_llm = LangChainInterface(
    model_id="mistralai/mistral-7b-instruct-v0-2",
    client=Client(credentials=Credentials.from_env()),
    parameters=TextGenerationParameters(
        decoding_method=DecodingMethod.SAMPLE,
        max_new_tokens=300,
        min_new_tokens=30,
        temperature=0.1,
        top_k=50,
        top_p=1,
    ),
    moderations=ModerationParameters(
        # Threshold is set to very low level to flag everything (testing purposes)
        # or set to True to enable HAP with default settings
        hap=ModerationHAP(input=True, output=True, threshold=0.01)
    ),
)
