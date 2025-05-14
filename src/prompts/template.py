import os
import re
from datetime import datetime
from typing import Dict, List

from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState

from src.prompts.coordinator import register_templates as register_coordinator_templates
from src.prompts.planner import register_templates as register_planner_templates
from src.prompts.supervisor import register_templates as register_supervisor_templates
from src.prompts.researcher import register_templates as register_researcher_templates
from src.prompts.coder import register_templates as register_coder_templates
from src.prompts.browser import register_templates as register_browser_templates
from src.prompts.reporter import register_templates as register_reporter_templates
from src.prompts.knowledge_extraction import register_templates as register_knowledge_templates
from src.prompts.entity_extraction import register_templates as register_entity_templates


class OpenManusPromptTemplate:
    """OpenManus prompt template manager for handling agent-specific prompts."""

    @staticmethod
    def get_prompt_template(prompt_name: str) -> str:
        """Load and process a prompt template from file.

        Args:
            prompt_name: Name of the prompt template file (without .md extension)

        Returns:
            Processed template string with variable placeholders
        """
        template_path = os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # Escape curly braces for string formatting
        template = template.replace("{", "{{").replace("}", "}}")
        # Convert <<VAR>> to {VAR} format
        template = re.sub(r"<<([^>>]+)>>", r"{1}", template)
        return template

    @staticmethod
    def apply_prompt_template(prompt_name: str, state: AgentState) -> List[Dict[str, str]]:
        """Apply a prompt template with current state variables.

        Args:
            prompt_name: Name of the prompt template to apply
            state: Current agent state containing variables and messages

        Returns:
            List of message dictionaries with system prompt and state messages
        """
        # Format current time in a consistent format
        current_time = datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")

        # Create and format the system prompt
        system_prompt = PromptTemplate(
            input_variables=["CURRENT_TIME"],
            template=OpenManusPromptTemplate.get_prompt_template(prompt_name),
        ).format(CURRENT_TIME=current_time, **state)

        # Combine system prompt with existing messages
        return [{"role": "system", "content": system_prompt}] + state["messages"]
        
    @classmethod
    def initialize(cls):
        """Initialize the prompt templates."""
        register_coordinator_templates(cls.templates)
        register_planner_templates(cls.templates)
        register_supervisor_templates(cls.templates)
        register_researcher_templates(cls.templates)
        register_coder_templates(cls.templates)
        register_browser_templates(cls.templates)
        register_reporter_templates(cls.templates)
        register_knowledge_templates(cls.templates)
        register_entity_templates(cls.templates)