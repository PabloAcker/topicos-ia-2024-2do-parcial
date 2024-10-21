from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent
from ai_assistant.tools import (
    travel_guide_tool,
    flight_tool,
    hotel_tool,
    bus_tool,
    restaurant_tool,
    trip_report_tool
)
from ai_assistant.prompts import agent_prompt_tpl, travel_guide_qa_tpl

class TravelAgent:
    def __init__(self, system_prompt: PromptTemplate | None = None):
        self.agent = ReActAgent.from_tools(
            [
                travel_guide_tool,
                flight_tool,
                hotel_tool,
                bus_tool,
                restaurant_tool,
                trip_report_tool
            ],
            verbose=True,
        )
        if system_prompt is not None:
            self.agent.update_prompts({"agent_worker:system_prompt": system_prompt})

    def get_agent(self) -> ReActAgent:
        return self.agent
