from typing import List, Dict, Any, Optional, Set
import random
from pydantic import BaseModel, Field, field_validator, model_validator
from dataclasses import dataclass, field
from llm.model.conversation_intents import (
    user_intent_to_id,
    USER_INTENTS,
    OPTIMIZABLE_USER_INTENTS,
)

@dataclass
class Location:
    longitude: float
    latitude: float

    def to_dict(self):
        return {
            "lng": self.longitude,
            "lat": self.latitude
        }
    
@dataclass
class LOS:
    title: Optional[str] = None
    location: Location = field(default_factory=lambda: Location(0.0, 0.0))
    address: Optional[str] = None
    opening_times: Optional[str] = None
    types: List[str] = field(default_factory=list)
    costs: Optional[str] = None  # Example: "20-30 EUR"
    ratings: Optional[float] = None  # Expected to be in the range 1–5
    foodtypes: Optional[List[str]] = field(default_factory=list)
    payments: List[str] = field(default_factory=list)
    distance: Optional[str] = None

    def to_dict(self):
        """Convert the Utterance object into a JSON-serializable dictionary."""
        return {
                "title": self.title,
                "location": {
                    "latitude": self.location.latitude,
                    "longitude": self.location.longitude,
                } if self.location else None,
                "address": self.address,
                "opening_times": self.opening_times,
                "types": self.types,
                "costs": self.costs,
                "ratings": self.ratings,
                "foodtypes": self.foodtypes if self.foodtypes is not None else [],
                "payments": self.payments,
                "distance": self.distance
            }
    

class Coordinates(BaseModel):
    lat: float
    lng: float

class ContentInput(BaseModel):
    pass


class ContentOutput(BaseModel):
    pass


class Utterance(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    seed: Optional[str] = None
 
    ordinal_vars: List[float] = field(default_factory=list) 
    categorical_vars: List[int] = field(default_factory=list)
 
    content_input: Optional[ContentInput] = None
    content_output_list: List[ContentOutput] = field(default_factory=list)
    raw_output: Any = None


class Turn(Utterance):
    """
    Represents a single turn in a conversation.
    """
    question_intent: Optional[str] = None
    answer_intent_classified: Optional[str] = None

    poi_exists: Optional[bool] = None
    user_intent_influences_fit: Optional[bool] = None

# NOTE: Turn is a complete user question & system answer pair. Utterance is either a user question or a system answer.
# NOTE: All conversations start with 'start' intent and end with 'stop' intent -> these intents are not part of the search space.
class Conversation(BaseModel):
    """
    Conversation-level variables for optimization:
    - ordinal_vars: style features (slang, politeness, etc.) - same as Utterance
    - categorical_vars: content features (category, payment_method, etc, num_turns)
    - intent_priorities: continuous values [0.001, 1.0] for each optimizable intent
    """
    _assigned_user_id: Optional[str] = None  # for Industry interaction

    turns: List[Turn] = Field(default_factory=list)
    seed: Optional[str] = None 
    
    # optimization variables
    ordinal_vars: List[float] = Field(default_factory=list)  # style features
    categorical_vars: List[int] = Field(default_factory=list)  # content + num_turns
    continuous_vars: List[float] = Field(default_factory=list)  # continuous priorities for each optimizable intent

    # specific values of variables
    style_input: Optional[Any] = Field(default=None)  # StyleDescription for the conversation
    content_input_values: Dict[str, Any] = Field(default_factory=dict)
    content_input_used: Set[str] = Field(default_factory=set)

    def __len__(self) -> int:
        return len(self.turns)
    
    def get_num_utterances(self) -> int:
        return sum(
            (1 if u.question else 0) + (1 if u.answer is not None else 0)
            for u in self.turns
        )
    
    @property
    def assigned_user_id(self) -> Optional[str]:
        return self._assigned_user_id

    @assigned_user_id.setter
    def assigned_user_id(self, value: str):
        self._assigned_user_id = value

    def get_dialogue_history_str(self) -> str:
        """Builds a single string containing dialogue history."""
        history = []
        for utterance in self.turns:
            if utterance.question:
                history.append(f"User: {utterance.question}")
            if utterance.answer:
                history.append(f"Assistant: {utterance.answer}")
        return "\n".join(history) if history else ""
    
    def get_dialogue_history_dict(self) -> Dict[str, Any]:
        """Builds a dictionary containing dialogue history."""
        history = {}
        for idx, utterance in enumerate(self.turns):
            turn_id = f"T{idx + 1}"
            history[turn_id] = {
                "question": utterance.question,
                "answer": utterance.answer
            }
        return history

    def get_intent_history_dict(self) -> Dict[str, Any]:
        """Builds a dictionary containing intent history."""
        history = {}
        for idx, utterance in enumerate(self.turns):
            turn_id = f"T{idx + 1}"
            history[turn_id] = {
                "question": utterance.question_intent,
                "answer": utterance.answer_intent_classified
            }
        return history

    def to_dict(self) -> Dict[str, Any]:
        turns_dict = {}

        for idx, turn in enumerate(self.turns):
            utterance_id = f"T{idx + 1}"

            serialized_content_input = (
                turn.content_input.model_dump() if getattr(turn, "content_input", None) is not None else None
            )

            serialized_content_output_list = [
                co.model_dump() if hasattr(co, "model_dump") else co
                for co in turn.content_output_list
            ]

            turns_dict[utterance_id] = {
                "question": turn.question,
                "answer": turn.answer,
                "question_intent": turn.question_intent,
                "answer_intent_classified": turn.answer_intent_classified,
                "content_input": serialized_content_input,
                "content_output": serialized_content_output_list,
            }

        return {
            "user_id": self.assigned_user_id,
            "num_turns": len(self.turns),
            "ordinal_vars": self.ordinal_vars,
            "categorical_vars": self.categorical_vars,
            "intent_priorities": self.continuous_vars,
            "content_input_values": self.content_input_values,
            "content_input_used": sorted(list(self.content_input_used)),
            "turns": turns_dict,
        }