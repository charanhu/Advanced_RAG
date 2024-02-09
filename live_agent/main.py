import os
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents.initialize import initialize_agent
from langchain.agents import Tool
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()




class AGENT:
    def __init__(self):
        self.llm = GoogleGenerativeAI(model="gemini-pro")

    def create_tool(self)

    
    def create_agent(self,  agent="zero-shot-react-description"):
        llm = self.llm
        wrapper = DuckDuckGoSearchAPIWrapper(time="d")
        # create a search agent
        search = DuckDuckGoSearchResults(api_wrapper=wrapper)
        tools = [search]
        agent = initialize_agent(
            agent=agent,
            tools=tools,
            llm=llm,
            verbose=True
        )
        
        return agent