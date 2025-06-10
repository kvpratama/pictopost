import json
import logging
import uuid
from graph import graph_memory

# Configure logger
logger = logging.getLogger(__name__)

class LangGraphLocalClient:
    # def __init__(self, google_api_key, tavily_api_key):
    def __init__(self):
        logger.info(f"Initializing LangGraphLocalClient")
        self.config = self.create_config()
        logger.debug(f"Client initialized with thread: {self.config}")
        
    def create_config(self):
        """Create a new thread with configurable parameters"""
        return {"configurable": 
                    {
                        "thread_id": str(uuid.uuid4()),
                        # "google_api_key": google_api_key,
                    }
                }
    
    def run_graph(self, input_data):
        """Run the graph with input data"""
        logger.info("Starting graph execution")
        logger.debug(f"Thread: {self.config}")
        logger.debug(f"Input data: {json.dumps(input_data, indent=2)}")
        response = graph_memory.invoke(input_data, self.config)
        return response
    
    def run_graph_resume(self, input_data):
        """Resume graph execution with updated input data"""
        logger.info("Resuming graph execution with updated input")
        logger.debug(f"Thread: {self.config}")
        logger.debug(f"Resume data: {json.dumps(input_data, indent=2)}")
        
        graph_memory.update_state(self.config, input_data)
        response = graph_memory.invoke(None, self.config)
        return response

    def run_graph_stream(self, input_data):
        """Run graph and stream the results"""
        logger.info("Starting graph stream execution")
        logger.debug(f"Thread: {self.config}")
        logger.debug(f"Input data: {json.dumps(input_data, indent=2)}")
        
        for event in graph_memory.stream(None, self.config, subgraphs=True, stream_mode="updates"):
            _, data = event  # event[1] → data
            if data.get('generate_question', ''):
                question = data.get('generate_question', '')["messages"][0].content
                logger.info(f"Question: {question}")
                yield question + "\n\n --- \n\n"
            if data.get('generate_answer', ''):
                answer = data.get('generate_answer', '')["messages"][0].content
                logger.info(f"Answer: {answer}")
                yield answer + "\n\n --- \n\n"
            if data.get('write_section', ''):
                section = data.get('write_section', '')["sections"][0]
                logger.info(f"Section: {section}")
                yield section + "\n\n --- \n\n"

    def get_state(self):
        """Get the current state of the thread"""
        state_data = graph_memory.get_state(self.config)[0]
        return state_data