import json
from typing import Callable, Dict, List, Union
import os
from enum import Enum, auto
from pydantic import BaseModel, Field
from src.llm.models import create_llm, generate
from src.config.logging import logger
from src.utils.io import read_file,write_to_file
from dotenv import load_dotenv
from src.tools.google import search as google_search
from src.tools.wiki import search as wiki_search
load_dotenv()

Observation = Union[str, Exception]

PROMPT_TEMPLATE_PATH = os.getenv("PROMPT_TEMPLATE_PATH")
OUTPUT_TRACE_PATH = os.getenv("OUTPUT_TRACE_PATH")

class Name(Enum):
    """
    Enumeration for tool names available to the agent.
    """
    WIKIPEDIA = auto()
    GOOGLE = auto()
    NONE = auto()

    def __str__(self) -> str:
        """
        String representation of the tool name.
        """
        return self.name.lower()

class Message(BaseModel):
    """
    Represents a message with sender role and content.
    """
    role: str = Field(..., description="The role of the message sender.")
    content: str = Field(..., description="The content of the message.")


class Tool:
    """
    A wrapper class for tools used by the agent, executing a function based on tool type.
    """

    def __init__(self, name: Name, func: Callable[[str], str]):
        """
        Initializes a Tool with a name and an associated function.
        
        Args:
            name (Name): The name of the tool.
            func (Callable[[str], str]): The function associated with the tool.
        """
        self.name = name
        self.func = func

    def use(self, query: str) -> Observation:
        """
        Executes the tool's function with the provided query.

        Args:
            query (str): The input query for the tool.

        Returns:
            Observation: Result of the tool's function or an error message if an exception occurs.
        """
        try:
            return self.func(query)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return str(e)


class Agent:
    """
    Defines the agent responsible for executing queries and handling tool interactions.
    """

    def __init__(self, model: str) -> None:
        """
        Initializes the Agent with a generative model, tools dictionary, and a messages log.

        Args:
            model (GenerativeModel): The generative model used by the agent.
        """
        logger.info("Initializing Agent...")
        self.model = create_llm(MODEL_KEY)
        self.tools: Dict[Name, Tool] = {}
        self.messages: List[Message] = []
        self.query = ""
        self.max_iterations = 5
        self.current_iteration = 0
        self.template = self.load_template()

    def load_template(self) -> str:
        """
        Loads the prompt template from a file.

        Returns:
            str: The content of the prompt template file.
        """
        return read_file(PROMPT_TEMPLATE_PATH)
    
    def register(self, name: Name, func: Callable[[str], str]) -> None:
        """
        Registers a tool to the agent.

        Args:
            name (Name): The name of the tool.
            func (Callable[[str], str]): The function associated with the tool.
        """
        self.tools[name] = Tool(name, func)
    
    def trace(self, role: str, content: str) -> None:
        """
        Logs the message with the specified role and content and writes to file.

        Args:
            role (str): The role of the message sender.
            content (str): The content of the message.
        """
        self.messages.append(Message(role=role, content=content))
        write_to_file(path=OUTPUT_TRACE_PATH, content=f"{role}: {content}\n")
    
    def get_history(self) -> str:
        """
        Retrieves the conversation history.

        Returns:
            str: Formatted history of messages.
        """
        return "\n".join([f"{message.role}: {message.content}" for message in self.messages])
    
    def ask_llm(self, prompt: str) -> str:
        """
        Queries the generative model with a prompt.

        Args:
            prompt (str): The prompt text for the model.

        Returns:
            str: The model's response as a string.
        """
        response = generate(
            model=self.model,
            prompt=prompt
        )
        return str(response) if response is not None else "No response from llm"
    
    def think(self) -> None:
        """
        Processes the current query, decides actions, and iterates until a solution or max iteration limit is reached.
        """
        self.current_iteration += 1
        logger.info(f"Starting iteration {self.current_iteration}")
        write_to_file(path=OUTPUT_TRACE_PATH, content=f"\n{'='*50}\nIteration {self.current_iteration}\n{'='*50}\n")

        if self.current_iteration > self.max_iterations:
            logger.warning("Reached maximum iterations. Stopping.")
            self.trace("assistant", "I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations. Here's what I know so far: " + self.get_history())
            return

        prompt = self.template.format(
            query=self.query, 
            history=self.get_history(),
            tools=', '.join([str(tool.name) for tool in self.tools.values()])
        )

        response = self.ask_llm(prompt)
        logger.info(f"Thinking => {response}")
        self.trace("assistant", f"Thought: {response}")
        self.decide(response)

    def decide(self, response: str) -> None:
        """
        Processes the agent's response, deciding actions or final answers.

        Args:
            response (str): The response generated by the model.
        """
        try:
            cleaned_response = response.strip().strip('`').strip()
            if cleaned_response.startswith('json'):
                cleaned_response = cleaned_response[4:].strip()
            
            parsed_response = json.loads(cleaned_response)
            
            if "action" in parsed_response:
                action = parsed_response["action"]
                tool_name = Name[action["name"].upper()]
                if tool_name == Name.NONE:
                    logger.info("No action needed. Proceeding to final answer.")
                    self.think()
                else:
                    self.trace("assistant", f"Action: Using {tool_name} tool")
                    self.act(tool_name, action.get("input", self.query))
            elif "answer" in parsed_response:
                self.trace("assistant", f"Final Answer: {parsed_response['answer']}")
            else:
                raise ValueError("Invalid response format")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {response}. Error: {str(e)}")
            self.trace("assistant", "I encountered an error in processing. Let me try again.")
            self.think()
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            self.trace("assistant", "I encountered an unexpected error. Let me try a different approach.")
            self.think()
        
    def act(self, tool_name: Name, query: str) -> None:
        """
        Executes the specified tool's function on the query and logs the result.

        Args:
            tool_name (Name): The tool to be used.
            query (str): The query for the tool.
        """
        tool = self.tools.get(tool_name)
        if tool:
            result = tool.use(query)
            observation = f"Observation from {tool_name}: {result}"
            self.trace("system", observation)
            self.think()
        else:
            logger.error(f"No tool registered for choice: {tool_name}")
            self.trace("system", f"Error: Tool {tool_name} not found")
            self.think()
    
    def execute(self, query: str) -> str:
        """
        Executes the agent's query-processing workflow.

        Args:
            query (str): The query to be processed.

        Returns:
            str: The final answer or last recorded message content.
        """
        self.query = query
        logger.info(f"Executing query: {query}")
        self.trace(role="user", content=query)
        self.think()
        return self.messages[-1].content
    
if __name__ == "__main__":
    MODEL_KEY = "together_llama"
    agent = Agent(model=MODEL_KEY)
    agent.register(Name.WIKIPEDIA, wiki_search)
    agent.register(Name.GOOGLE, google_search)
    query = "What is the age of the oldest player in the country that has won the most FIFA World Cup titles?"
    answer = agent.execute(query)
    logger.info(f"Final Answer: {answer}")
