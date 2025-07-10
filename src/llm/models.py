from src.config.logging import logger
import os
import json
from langchain_openai import ChatOpenAI
from typing import Optional


def create_llm(model: str) -> ChatOpenAI:
    """
    Initializes a Together.ai-compatible ChatOpenAI model using environment config.

    Args:
        model (str): The model key used in the .env variable

    Returns:
        ChatOpenAI: Configured LangChain LLM instance
    """
    try:
        env_key = f"LLM_MODEL_CONFIG_{model}"
        env_value = os.environ.get(env_key)

        if not env_value:
            raise ValueError(f"Missing environment variable: {env_key}")

        config = json.loads(env_value)
        api_key = config.get("api_key")
        api_endpoint = config.get("api_endpoint")
        model_name = config.get("model")

        if not all([api_key, api_endpoint, model_name]):
            raise ValueError(f"Incomplete configuration in {env_key}")

        return ChatOpenAI(
            api_key=api_key,
            base_url=api_endpoint,
            model=model_name,
            temperature=0.0,
        )

    except Exception as e:
        logger.error(f"Error initializing LLM for model '{model}': {e}")
        raise


def generate(model: ChatOpenAI, prompt: str) -> Optional[str]:
    """
    Generates a response from a Together.ai model using LangChain.

    Args:
        model (ChatOpenAI): Configured LangChain LLM instance
        prompt (str): The prompt string to send to the model

    Returns:
        Optional[str]: Generated response content
    """
    try:
        logger.info(f"Generating response with model '{getattr(model, 'model_name', str(model))}'")
        response = model.invoke(prompt)

        if not getattr(response, "content", None):
            logger.error("Empty response from model")
            return None

        logger.info("Successfully generated response")
        return response.content
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return None



if __name__ == "__main__":
    # Load .env variables
    from dotenv import load_dotenv
    load_dotenv()
    MODEL_KEY = "together_llama"
    model = create_llm(MODEL_KEY)
    result = generate(
        model=model,
        prompt="What's the largest planet in the solar system?"
    )
    print(result)
