from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
# tools
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, MetaData
from llama_index.core import SQLDatabase
from llama_index.core.tools import QueryEngineTool, BaseTool, FunctionTool
# search
import requests
import os
import asyncio
import json
import time
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from requests.exceptions import HTTPError, RequestException
from llama_index.core import PromptTemplate
# agent
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer


Settings.llm = Ollama(model="qwen2.5-coder:7b", base_url="http://127.0.0.1:11434")
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")


def get_sql_agent():
    with open("table_laptop.sql", "r") as table_schema_file:
        table_schemas = table_schema_file.read()
    # Connect to the existing database
    engine = create_engine("sqlite:///laptop-price-spesification.db")

    # Create a MetaData object
    metadata = MetaData()

    # Reflect the existing database schema
    metadata.reflect(bind=engine)

    # Creating Query Engine
    sql_database = SQLDatabase(engine, include_tables=["laptops"])
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["laptops"],
        #llm=llm,
    )

    # Creating Tool
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "A tool designed to translate natural language queries into SQL commands, "
            "specifically for interacting with a database table containing information about laptops. "
            "This tool simplifies complex SQL query generation and provides precise data retrieval.\n\n"
            "Table Schema:\n"
            f"{table_schemas}\n\n"
            "Important Notes:\n"
            "- Always generate full SELECT queries, selecting all columns.\n"
            "- Ensure searches align with existing table columns.\n"
            "- Perform searches in a case-insensitive manner.\n"
            "- Always answer only based on data that exists in the database.\n"
            "- If no data is found, respond with: 'No data found in the database for the given query.'"
        ),
        name="sql_tool"
    )
    return sql_tool



def scrape_page(url):
    print("Scraping page...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise HTTPError for bad status codes
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else "No title found"
        paragraphs = soup.find_all("p")
        content = "\n".join(p.get_text() for p in paragraphs)
        return {"title": title, "content": content}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {"error": str(e)}

def search_internet(keyword: str) -> str:
    output = f"# Internet Search Result for '{keyword}' text \n"
    print("Searching...", keyword)
    try:
        search_query = DDGS().text(keyword, region="id-id", max_results=4)
    
        for result in search_query:
            print("masuk ", result['title'])
            title = result["title"]
            href = result["href"]
            body = result["body"]

            scrape_result = scrape_page(href)

            result = {
                "Type": "Webpage",
                "Title": title,
                "Link": href,
                "Body": body,
                "Detail": scrape_result,
            }
            # Add search results to output
            output += json.dumps(result)
            # save_to_index(result, title, href)
            print(output)

        return output  # Return output when the search completes successfully
    except Exception as e:
        # General exception catch
        print(f"An unexpected error occurred: {e}")

    return output  # Return output after retries (if any)

search_tool = FunctionTool.from_defaults(
    search_internet, 
    description= """
    A tool that performs internet searches to retrieve relevant information based on user queries. 
    It connects to web resources, processes results, and presents concise and accurate information.
    The tool is designed to assist in retrieving up-to-date and contextually appropriate data for a variety of topics.
    ALWAYS provide a link as a source for your answer
    """
)

sql_tool = get_sql_agent()
memory = ChatMemoryBuffer.from_defaults(token_limit=32768)
# setup ReAct Agent 
agent = ReActAgent.from_tools(
    [search_tool,sql_tool],
    # llm=llm,
    verbose=True,
    # context=context
    memory=memory,
)

react_system_header_str = """\

You are designed to assist with a wide range of laptop-related inquiries, including helping users compare models, providing detailed specifications, recommending options based on specific preferences, and offering expert advice on the best laptop choices for various needs and budgets. If a search in the local database does not return relevant results, you will then search the internet for the information. If the internet search still doesn't yield the requested information, you will inform the user that you couldn't retrieve the information, explaining the reason why it couldn't be found. When using the internet for searches, always include a link to the source of your information.

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Additional Rules
- The answer MUST contain a sequence of bullet points that explain how you arrived at the answer. This can include aspects of the previous conversation history.
- You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""
react_system_prompt = PromptTemplate(react_system_header_str)


agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})