from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI  # Example LLM
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv("../../.env")

# Initialize Langfuse client with constructor arguments
Langfuse(
    public_key="pk-lf-bae3dcd6-1356-4f30-87c7-0c104c50d596",
    secret_key="sk-lf-74281145-630f-49fd-868f-b66d7923a729",
    host="http://localhost:3000/"  # Optional: defaults to https://cloud.langfuse.com
)

# Get the configured client instance
langfuse = get_client()

# Initialize the Langfuse handler
langfuse_handler = CallbackHandler()

# Create your LangChain components
llm = ChatOpenAI(model_name="gpt-4.1")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm

# Run your chain with Langfuse tracing
response = chain.invoke({"topic": "cats"}, config={"callbacks": [langfuse_handler]})
print(response.content)

# Flush events to Langfuse in short-lived applications
langfuse.flush()