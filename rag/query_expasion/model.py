from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm_1 = GoogleGenerativeAI(model="gemini-pro",
                           temperature=0.0,
                           cache=False,
                           )
