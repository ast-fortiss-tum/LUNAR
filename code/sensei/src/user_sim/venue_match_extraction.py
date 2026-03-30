import json
import logging
import os
import time

from dotenv import load_dotenv
from json_repair import repair_json
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

logger = logging.getLogger("Info Logger")


class VenueMatchExtraction:
    """
    Detect whether the ASSISTANT proposed a specific venue that matches the USER's preferences,
    using ONLY the conversation history.
    All preferences of the user have to be matched by the assistant's proposal for a "match=true", otherwise "match=false".

    Conversation format expected (matches your UserGeneration.get_history()):
      'Assistant: ...\\nUser: ...\\nAssistant: ...'
    """

    def __init__(self, conversation_history):
        # Convert your structured history into the same string format UserGeneration.get_history() produces.
        # This avoids the "['{...}']" representation you get from f"{list}".
        self.conversation_text = self._history_to_text(conversation_history)

        self.prompt = """
You analyze a conversation transcript formatted as lines:
"User: ...", "Assistant: ..."

Goal:
Determine whether the Assistant has provided at least one specific venue/provider that suits the User's
preferences stated in the conversation.

Rules (critical):
- Only count preferences stated by the USER (lines starting with "User:").
- Only count venues proposed/confirmed by the ASSISTANT (lines starting with "Assistant:").
- A venue must be specific/identifiable (name and/or address/branch details). Generic suggestions like
  "try a nearby restaurant" do NOT count.
- The venue type (restaurant, car repair shop, etc.) and preferences are implied from the conversation.
- If the user has no preferences/constraints at all, set match=false unless the user explicitly says "any is fine".

Return JSON only:
{
  "match": true/false,
  "venue_type": string or null,
  "venue_name": string or null,
  "venue_details": string or null,
  "user_preferences": [string, ...],
  "matched_preferences": [string, ...],
  "unmet_preferences": [string, ...],
  "evidence": {
    "assistant_quote": string or null,
    "user_preferences_quote": string or null
  }
}
""".strip()

        self.messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": self.conversation_text},
        ]

    @staticmethod
    def _history_to_text(conversation_history) -> str:
        """
        conversation_history is expected to look like:
          {"interaction": [{"Assistant": "..."}, {"User": "..."} , ...]}
        """
        interaction = conversation_history.get("interaction", [])
        lines = []
        for turn in interaction:
            if not isinstance(turn, dict) or len(turn) != 1:
                continue
            role, msg = next(iter(turn.items()))
            lines.append(f"{role}: {msg}")
        return "\n".join(lines)

    @staticmethod
    def _as_bool(x) -> bool:
        if isinstance(x, bool):
            return x
        if x is None:
            return False
        if isinstance(x, str):
            return x.strip().lower() in ("true", "yes", "1")
        return bool(x)

    def detect(self, model="gpt-5-chat", max_retries=3, retry_delay=1):
        """
        Returns a dict matching the schema in the prompt.
        If max retries are reached (API failure, invalid JSON, etc.), returns a fallback with match=false.
        """
        last_err = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=self.messages,
                )

                raw = resp.choices[0].message.content if resp and resp.choices else ""
                fixed = repair_json(raw or "")
                payload = json.loads(fixed)

                result = {
                    "match": self._as_bool(payload.get("match")),
                    "venue_type": payload.get("venue_type"),
                    "venue_name": payload.get("venue_name"),
                    "venue_details": payload.get("venue_details"),
                    "user_preferences": payload.get("user_preferences") or [],
                    "matched_preferences": payload.get("matched_preferences") or [],
                    "unmet_preferences": payload.get("unmet_preferences") or [],
                    "evidence": payload.get("evidence") or {
                        "assistant_quote": None,
                        "user_preferences_quote": None,
                    },
                }

                # Hard guardrail: don't allow match=true without a concrete venue
                if not result["venue_name"]:
                    result["match"] = False

                # Extra guardrails: ensure lists are lists, evidence has required keys
                if not isinstance(result["user_preferences"], list):
                    result["user_preferences"] = []
                if not isinstance(result["matched_preferences"], list):
                    result["matched_preferences"] = []
                if not isinstance(result["unmet_preferences"], list):
                    result["unmet_preferences"] = []
                if not isinstance(result["evidence"], dict):
                    result["evidence"] = {"assistant_quote": None, "user_preferences_quote": None}
                result["evidence"].setdefault("assistant_quote", None)
                result["evidence"].setdefault("user_preferences_quote", None)

                return result

            except Exception as e:
                last_err = e
                logger.warning(f"VenueMatchExtraction attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)

        logger.error(f"VenueMatchExtraction: max retries reached. Returning fallback match=false. Last error: {last_err}")

        # Fallback: strict schema (no extra keys)
        return {
            "match": False,
            "venue_type": None,
            "venue_name": None,
            "venue_details": None,
            "user_preferences": [],
            "matched_preferences": [],
            "unmet_preferences": [],
            "evidence": {
                "assistant_quote": None,
                "user_preferences_quote": None,
            },
        }