from langchain_ollama import OllamaLLM
from langchain.schema import AIMessage
import logging

class GemmaModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = OllamaLLM(model="gemma:7b-instruct")
        
    def invoke(self, prompt):
        try:
            self.logger.info("Sending prompt to Gemma model via Ollama")
            response = self.llm.invoke(prompt)
            return AIMessage(content=response)
        except Exception as e:
            self.logger.error(f"Error invoking Gemma model: {e}")
            return AIMessage(content=f"Error generating response: {str(e)}")