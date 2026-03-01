"""
Agentic Medical Consult Team — AutoGen-powered multi-agent collaboration.

Architecture
────────────
User Query ──► SelectorGroupChat Team
                        ├── CMO Agent  (orchestrator / final answer)
                        ├── Researcher Agent  (has RAG tool access)
                        └── Safety Reviewer Agent  (safety gate)

The Researcher has direct access to tools (`search_medical_database`) that
wrap the existing LangChain → Pinecone → DSPy retrieval pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.autogen_tools import (
    search_medical_database,
    search_medical_database_multi,
    set_retriever as set_tool_retriever,
)
from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger("autogen_consult")

# ─── System Prompts ───────────────────────────────────────────────────────────

CMO_SYSTEM_PROMPT = """\
You are the **Chief Medical Officer (CMO)** in a collaborative AI medical consult team.

Your responsibilities:
1. Receive the patient's symptoms / query.
2. Decide what medical research is needed and delegate to the Medical Researcher.
3. Synthesize retrieved evidence into a clear, evidence-based assessment.
4. Formulate the final response covering: possible conditions, recommended next steps,
   and when to seek emergency care.
5. After the Safety Reviewer has approved (or flagged concerns), produce the **final
   answer** to the patient, incorporating any safety notes.

Communication rules:
- Always wait for research results before making a diagnosis.
- After the Safety Reviewer responds, write a message starting with
  "FINAL ANSWER:" that contains the consolidated response for the patient.
- Be thorough but concise. Use medical terminology with plain-English explanations.
- Always include a disclaimer that this is AI-generated information and not a
  substitute for professional medical advice.
"""

RESEARCHER_SYSTEM_PROMPT = """\
You are the **Medical Researcher** in a collaborative AI medical consult team.

Your responsibilities:
1. When the CMO requests research, use your tools to retrieve relevant medical 
   literature and guidelines from the knowledge base.
2. You may use the multi-query tool when multiple related queries would benefit 
   from parallel lookup with deduplication.
3. Summarize the retrieved evidence clearly, citing source documents.
4. If the initial search is insufficient, refine the query and search again.

Communication rules:
- Only speak when you have research results to share.
- Present findings in a structured format (numbered points).
- Do NOT make diagnoses — leave clinical reasoning to the CMO.
"""

SAFETY_REVIEWER_SYSTEM_PROMPT = """\
You are the **Safety Reviewer** in a collaborative AI medical consult team.

Your responsibilities:
1. Review the CMO's proposed assessment for safety concerns.
2. Flag any dangerous drug interactions, contraindications, missing red-flag
   symptoms, or situations that require immediate emergency care.
3. Verify that appropriate disclaimers are included.
4. If everything looks safe, respond with "SAFETY APPROVED" and any minor notes.
   If there are critical concerns, clearly state "SAFETY CONCERN:" followed by details.

Communication rules:
- Be concise and specific about safety issues.
- Always check for: allergies not addressed, emergency symptoms (chest pain,
  difficulty breathing, stroke signs), and dosage concerns.
- Do NOT rewrite the full response — only flag issues for the CMO to address.
"""


# ─── Agent Factory ────────────────────────────────────────────────────────────

class MedicalConsultTeam:
    """
    Encapsulates the AutoGen v0.4 consult team and exposes an
    async `arun(user_query)` interface for the API layer.
    """

    def __init__(self, retriever=None) -> None:
        # Inject the already-initialized retriever into the tool layer
        if retriever is not None:
            set_tool_retriever(retriever)

        settings = get_settings()
        
        # Strip "openai/" prefix if DSPy uses it
        model_name = settings.llm_model
        if model_name.startswith("openai/"):
            model_name = model_name[len("openai/"):]

        # Create v0.4 Model Client
        self.model_client = OpenAIChatCompletionClient(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
        )

        self._build_team()

    def _build_team(self) -> None:
        # 1. Chief Medical Officer
        self.cmo_agent = AssistantAgent(
            name="Chief_Medical_Officer",
            system_message=CMO_SYSTEM_PROMPT,
            model_client=self.model_client,
            description="The lead doctor who orchestrates research and gives the final diagnosis.",
        )

        # 2. Medical Researcher (Give tools directly to the agent in v0.4)
        self.researcher_agent = AssistantAgent(
            name="Medical_Researcher",
            system_message=RESEARCHER_SYSTEM_PROMPT,
            model_client=self.model_client,
            tools=[search_medical_database, search_medical_database_multi],
            description="A medical database researcher. Call this agent to look up medical information.",
        )

        # 3. Safety Reviewer
        self.safety_reviewer = AssistantAgent(
            name="Safety_Reviewer",
            system_message=SAFETY_REVIEWER_SYSTEM_PROMPT,
            model_client=self.model_client,
            description="Reviews diagnoses for safety, red flags, and contraindications.",
        )

        # 4. Define Termination Conditions
        # Stop when CMO says FINAL ANSWER, or after 12 max messages
        text_termination = TextMentionTermination("FINAL ANSWER:")
        max_messages = MaxMessageTermination(12)
        termination = text_termination | max_messages

        # 5. Create the SelectorGroupChat (Replaces GroupChatManager)
        self.team = SelectorGroupChat(
            participants=[self.cmo_agent, self.researcher_agent, self.safety_reviewer],
            model_client=self.model_client,
            termination_condition=termination,
            allow_repeated_speaker=False,
        )
        
        logger.info("AutoGen v0.4 Medical Consult Team Initialized.")

    # ── Public API ────────────────────────────────────────────────────────

    async def arun(self, user_query: str) -> dict[str, Any]:
        """
        Initiate a full multi-agent medical consultation asynchronously.
        """
        logger.info(f"Starting AutoGen consult for: {user_query[:100]}...")

        # Run the team task
        result = await self.team.run(task=user_query)

        # Extract results from the TaskResult object
        chat_history = []
        for msg in result.messages:
            # v0.4 message structure
            agent_name = getattr(msg, "source", "unknown")
            # Handle standard text messages vs tool call messages
            content = getattr(msg, "content", str(msg))
            if not isinstance(content, str):
                content = str(content)
            
            chat_history.append({
                "agent": agent_name,
                "content": content
            })

        final_answer = self._extract_final_answer(chat_history)
        rounds = len(chat_history)

        logger.info(f"AutoGen consult completed in {rounds} rounds.")

        return {
            "final_answer": final_answer,
            "chat_history": chat_history,
            "rounds": rounds,
        }

    def run(self, user_query: str) -> dict[str, Any]:
        """Sync wrapper for arun if needed in non-async contexts."""
        return asyncio.run(self.arun(user_query))

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_final_answer(messages: list[dict]) -> str:
        """
        Walk the chat history in reverse to find the CMO's final answer.
        """
        for msg in reversed(messages):
            content = msg.get("content", "")
            if "FINAL ANSWER:" in content:
                idx = content.index("FINAL ANSWER:")
                return content[idx + len("FINAL ANSWER:"):].strip()

        for msg in reversed(messages):
            content = msg.get("content", "")
            agent = msg.get("agent", "")
            if content and agent == "Chief_Medical_Officer":
                return content.strip()

        return "The medical consult team was unable to produce a final answer."