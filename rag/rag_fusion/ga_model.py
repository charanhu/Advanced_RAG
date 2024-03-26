# Import necessary modules from the IBM Watson Machine Learning SDK
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)
from ibm_watson_machine_learning.foundation_models.utils.enums import (
    ModelTypes,
    DecodingMethods,
)
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


# Import other required libraries
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# Load the environment variables
load_dotenv()
watsonx_api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)

# Set up the credentials for accessing IBM Watson
creds = {"url": ibm_cloud_url, "apikey": watsonx_api_key}

# Define the parameters for text generation
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.15,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
    GenParams.RANDOM_SEED: 999,
    GenParams.STOP_SEQUENCES: None,
    GenParams.REPETITION_PENALTY: 1.0,
}

# Set up the LLAMA2 model with the specified parameters and credentials
LLAMA2_MODEL = Model(
    model_id="meta-llama/llama-2-70b-chat",
    credentials=creds,
    project_id=project_id,
    params=parameters,
)

# Create a Watson LLM instance with the LLAMA2 model
watsonx_llm = WatsonxLLM(model=LLAMA2_MODEL)
