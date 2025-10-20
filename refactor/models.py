#!/usr/bin/env python3
# Concrete implementations of KeywordGenModel

import json
from typing import Any, Union, cast
import httpx
from .types import KeywordGenModel, ErrorReport
from ai import AIModel, AIMessage, ai_call

# Single unified prompt for all models
SYSTEM_PROMPT: str = """You are a search query generator for Slack. Given a user's natural language query, generate a list of 1-5 keyword search queries that would help find relevant Slack messages. Each search query should be 1-3 keywords that someone might use in Slack's search box.

Output format: JSON array of strings like ["query1", "query2", "query3"]"""

# Longer set of examples for all models
FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "user": "Why our billing has sudden increase when no traffic but robots.txt requests wake up GPU route?",
        "assistant": '["billing spike", "robots.txt requests", "GPU route"]'
    },
    {
        "user": "What's the recommended pattern for App-to-Sandbox relationships in our infrastructure, and what are the lifecycle implications when terminating Sandboxes?",
        "assistant": '["App Sandbox anti-pattern", "sandbox lifecycle", "terminate sandbox"]'
    },
    {
        "user": "how do i stop libmodal sandboxes bc terminate() isnt working and apps keep running",
        "assistant": '["libmodal sandbox terminate", "stop libmodal sandboxes", "sandbox app running"]'
    },
    {
        "user": "Hey folks, quick question - what's the best way to handle results from long-running Modal jobs? Should I be polling for them or is there a better pattern?",
        "assistant": '["Modal Queues", "Modal job results", "polling pattern", "long-running jobs"]'
    },
    {
        "user": "why does my node script stop running after my api sends the response back",
        "assistant": '["node script stops", "api response", "web server subprocess", "container shutdown"]'
    },
    {
        "user": "Why do my background processes keep getting cut off before they finish running?",
        "assistant": '["background process", "container scaledown", "async work queue", "nodejs subprocess"]'
    }
]

def _build_messages(query: str) -> list[AIMessage]:
    messages: list[AIMessage] = [AIMessage(role="system", content=SYSTEM_PROMPT)]

    for example in FEW_SHOT_EXAMPLES:
        messages.append(AIMessage(role="user", content=example["user"]))
        messages.append(AIMessage(role="assistant", content=example["assistant"]))

    # Add the actual query
    messages.append(AIMessage(role="user", content=query))

    return messages

def _parse_json_response(response: str) -> list[str] | ErrorReport:
    try:
        result: Any = json.loads(response)
        if isinstance(result, list):
            return cast(list[str], result)
        else:
            return ErrorReport(message=f"JSON format is unexpected: {response}")
    except json.JSONDecodeError:
        return ErrorReport(message=f"Response is not a JSON array: {response}")


class OpenAI(KeywordGenModel):
    """OpenAI models using ai.py interface"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        super().__init__(model_name=f"openai/{model}")
        self.ai_model: AIModel = AIModel(company="openai", model=model)
        self.temperature: float = temperature

    async def generate_search_strings(self, query: str) -> list[str] | ErrorReport:
        try:
            messages: list[AIMessage] = _build_messages(query)

            response: str = await ai_call(
                model=self.ai_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=150
            )

            return _parse_json_response(response)

        except Exception as e:
            return ErrorReport(message=f"OpenAI API error: {str(e)}")


class LocalVLLMModel(KeywordGenModel):
    """Local vLLM model that calls OpenAI-like API on localhost"""

    def __init__(self, model_id: str, port: int = 8000, temperature: float = 0.0) -> None:
        super().__init__(model_name=f"local/{model_id}")
        self.temperature: float = temperature
        self.port: int = port

    async def generate_search_strings(self, query: str) -> Union[list[str], ErrorReport]:
        try:
            # Build messages for chat completions API
            messages: list[dict[str, str]] = []
            messages.append({"role": "system", "content": SYSTEM_PROMPT})

            # Add few-shot examples
            for example in FEW_SHOT_EXAMPLES:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

            # Add the actual query
            messages.append({"role": "user", "content": query})

            # Call local chat completions API directly
            async with httpx.AsyncClient() as client:
                response: httpx.Response = await client.post(
                    f"http://localhost:{self.port}/v1/chat/completions",
                    json={
                        "messages": messages
                    },
                    timeout=30.0
                )

                if response.status_code != 200:
                    return ErrorReport(message=f"Local API error {response.status_code}: {response.text}")

                result_data: dict[str, Any] = response.json()
                content: str = result_data["choices"][0]["message"]["content"]

                return _parse_json_response(content)

        except Exception as e:
            return ErrorReport(message=f"Local vLLM API error: {str(e)}")
