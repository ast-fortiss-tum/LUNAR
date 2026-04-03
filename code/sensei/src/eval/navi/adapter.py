from __future__ import annotations

import json
import os
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from user_sim.data_extraction import DataExtraction
from user_sim.utils.config import errors

import sys
sys.path.insert(0, "../lunar/venv/lib/python3.11/site-packages")
sys.path.insert(0, "../lunar/")

from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.model.models import Conversation as LLMConversation
from llm.model.models import Turn as LLMTurn

# Content models (adjust if your paths differ)
from eval.navi.models import NaviContentInput, NaviContentOutput
from llm.model.models import Coordinates
from eval.navi.fitness import get_fitness_fnc, get_critical_fnc
from eval.navi.fitness import get_fitness_fnc_carcontrol, get_critical_fnc_carcontrol

from llm.llms import LLMType

import numpy as np
import wandb
import os

from llm.llms import ModelStatistics

def write_token_usage(save_folder):
    with open(save_folder + os.sep + "llm_usage_summary.json", "w") as f:
        usage_summary = ModelStatistics.complete_statistics()
        usage_summary["total_tokens"] = ModelStatistics.total_values()
        json.dump(usage_summary, f, indent=4)
        if wandb.run is not None:
            wandb.log(usage_summary)

def get_conversation_metadata(user_profile, the_user, serial=None):
    def conversation_metadata(up):
        interaction_style_list = []
        conversation_list = []

        for inter in up.interaction_styles:
            interaction_style_list.append(inter.get_metadata())

        conversation_list.append({"interaction_style": interaction_style_list})

        if isinstance(up.yaml["conversation"]["number"], int):
            conversation_list.append({"number": up.yaml["conversation"]["number"]})
        else:
            conversation_list.append({"number": up.conversation_number})

        if "random steps" in up.yaml["conversation"]["goal_style"]:
            conversation_list.append({"goal_style": {"steps": up.goal_style[1]}})
        else:
            conversation_list.append(up.yaml["conversation"]["goal_style"])

        return conversation_list

    def ask_about_metadata(up):
        if not up.ask_about.variable_list:
            return up.ask_about.str_list
        return user_profile.ask_about.str_list + user_profile.ask_about.picked_elements

    def data_output_extraction(u_profile, user):
        output_list = u_profile.output
        data_list = []
        for output in output_list:
            var_name = list(output.keys())[0]
            var_dict = output.get(var_name)
            my_data_extract = DataExtraction(
                user.conversation_history,
                var_name,
                var_dict["type"],
                var_dict["description"],
            )
            data_list.append(my_data_extract.get_data_extraction())

        data_dict = {k: v for dic in data_list for k, v in dic.items()}
        has_none = any(value is None for value in data_dict.values())
        if has_none:
            count_none = sum(1 for value in data_dict.values() if value is None)
            errors.append({1001: f"{count_none} goals left to complete."})

        return data_list

    data_output = {"data_output": data_output_extraction(user_profile, the_user)}
    context = {"context": user_profile.raw_context}
    ask_about = {"ask_about": ask_about_metadata(user_profile)}
    conversation = {"conversation": conversation_metadata(user_profile)}
    language = {"language": user_profile.language}
    serial_dict = {"serial": serial}
    errors_dict = {"errors": errors}
    variables_per_turn = {"variables_per_turn": the_user.variables_per_turn}

    metadata = {
        **serial_dict,
        **language,
        **context,
        **ask_about,
        **conversation,
        **data_output,
        **errors_dict,
        **variables_per_turn,
    }
    return metadata


# -------------------------
# history -> (question, answer) turns
# -------------------------

def _extract_interaction_list(conversation_history: Any) -> List[Dict[str, Any]]:
    """
    Your UserGeneration.update_history stores like:
      conversation_history = {"interaction": [ {"User": "..."}, {"Assistant": "..."} ], ...}

    This returns that list.
    """
    if isinstance(conversation_history, dict):
        interaction = conversation_history.get("interaction", [])
        return interaction if isinstance(interaction, list) else []
    if isinstance(conversation_history, list):
        return conversation_history
    return []


def _pair_history_as_turns(interaction_list: Any) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Expects: [{"User": "..."}, {"Assistant": "..."}, ...]
    Returns: [(question, answer), ...]
    """
    if not isinstance(interaction_list, list):
        return []

    turns: List[Tuple[Optional[str], Optional[str]]] = []
    pending_user: Optional[str] = None

    for it in interaction_list:
        if not (isinstance(it, dict) and len(it) == 1):
            continue
        role, text = next(iter(it.items()))
        if role not in ("User", "Assistant"):
            # ignore stray dicts
            continue

        if text is not None and not isinstance(text, str):
            text = str(text)

        if role == "User":
            if pending_user is not None:
                turns.append((pending_user, None))
            pending_user = text
        else:  # Assistant
            if pending_user is None:
                continue
            turns.append((pending_user, text))
            pending_user = None

    if pending_user is not None:
        turns.append((pending_user, None))

    return turns


# -------------------------
# variables_per_turn -> NaviContentInput
# retrieved_objs_per_turn -> List[NaviContentOutput]
# -------------------------

def _first(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, list):
        return x[0] if x else None
    return x


def _to_coordinates(lat: Any, lng: Any) -> Optional[Coordinates]:
    try:
        if lat is None or lng is None:
            return None
        return Coordinates(lat=float(lat), lng=float(lng))
    except Exception:
        return None


def _normalize_yes_no(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return "yes"
    if s in ("no", "n", "false", "0", "noo"):
        return "no"
    return str(x)


def _map_variables_to_content_input(vars_for_turn: Any) -> Optional[NaviContentInput]:
    """
    Example vars_for_turn:
      {'poi_type': ['restaurant'], 'payment_method': ['CREDIT_CARD'], 'rating': [4.5], 'parking_yes': ['NOO']}
    """
    if not isinstance(vars_for_turn, dict):
        return None

    poi_type = _first(vars_for_turn.get("poi_type"))
    payment_method = _first(vars_for_turn.get("payment_method"))
    rating = _first(vars_for_turn.get("rating"))

    return NaviContentInput(
        title=None,
        category=str(poi_type) if poi_type is not None else None,
        address=None,
        location=None,
        business_hours_status=None,
        payment_method=str(payment_method) if payment_method is not None else None,
        rating=float(rating) if isinstance(rating, (int, float)) else None,
        price_range=_first(vars_for_turn.get("price_range") or vars_for_turn.get("price_level")),
        fuel_price=_first(vars_for_turn.get("fuel_price")),
        fuel_type=_first(vars_for_turn.get("fuel_type")),
        gas_station_brand=_first(vars_for_turn.get("gas_station_brand")),
        restaurant_brand=_first(vars_for_turn.get("restaurant_brand")),
        food_type=_first(vars_for_turn.get("food_type")),
        parking=_normalize_yes_no(_first(vars_for_turn.get("parking_yes") or vars_for_turn.get("parking"))),
        charging=_first(vars_for_turn.get("charging")),
        availability=_first(vars_for_turn.get("availability")),
    )


def _split_categories(cat: Any) -> Optional[List[str]]:
    if not isinstance(cat, str) or not cat.strip():
        return None
    parts = [c.strip() for c in cat.split(",") if c.strip()]
    return parts or None


def _map_retrieved_to_content_output_list(retrieved_for_turn: Any) -> List[NaviContentOutput]:
    """
    Example retrieved_for_turn:
      [{'name': 'Coffee House Too Cafe', 'category': 'Restaurants, Cafes', 'rating': 4.5, ...}, ...]
    """
    if not isinstance(retrieved_for_turn, list):
        return []

    outs: List[NaviContentOutput] = []
    for poi in retrieved_for_turn:
        if not isinstance(poi, dict):
            continue

        outs.append(
            NaviContentOutput(
                title=poi.get("name"),
                categories=_split_categories(poi.get("category")),
                address=poi.get("address"),
                location=_to_coordinates(poi.get("latitude"), poi.get("longitude")),
                business_hours_status=poi.get("business_hours_status"),
                payment_methods=poi.get("payment_methods"),
                rating=poi.get("rating"),
                price_range=poi.get("price_level") or poi.get("price_range"),
                fuel_prices=poi.get("fuel_prices"),
                fuel_types=poi.get("fuel_types"),
                gas_station_brand=poi.get("gas_station_brand"),
                restaurant_brand=poi.get("restaurant_brand"),
                food_types=poi.get("food_types"),
                parking=("yes" if poi.get("parking") is True else ("no" if poi.get("parking") is False else None)),
                charging=poi.get("charging"),
                availability=poi.get("availability"),
            )
        )
    return outs

def evaluate_simout(simout, args):
    if "car" in args.user:
        print("car control case study")
        fitness_fnc = get_fitness_fnc_carcontrol(llm_type=LLMType(args.judge_llm), 
                                weights=[args.weight_clarity, args.weight_request_orientedness],
                                dimension_labels=["Clarity", "Request-Orientedness"], 
                                max_score=2)
    else:
        print("navi case study")
        fitness_fnc = get_fitness_fnc(llm_type=LLMType(args.judge_llm), 
                                    weights=[args.weight_clarity, args.weight_request_orientedness],
                                    dimension_labels=["Clarity", "Request-Orientedness"], 
                                    max_score=2)
    fitness = fitness_fnc.eval(simout)
    print("fitness", fitness)
    vector_fitness = np.array(fitness)
    critical_fnc = get_critical_fnc(fitness_fnc, score_threshold=args.critical_threshold)
    critical = critical_fnc.eval(vector_fitness, simout = simout)
    print("critical", critical)
    result = {
        "fitness_scores" : {}
    }
    for fitness_name, fitness_score in zip(fitness_fnc.name, fitness):
        result["fitness_scores"][fitness_name] = fitness_score
    result["is_critical"] = critical
    return result   
# -------------------------
# main converter
# -------------------------

def convert_to_simout(
    user_profile: Any,
    the_user: Any,
    serial: Optional[str] = None,
    model: Optional[str] = None,
    ipa: str = "",
    time_conv: Optional[float] = None,
) -> MultiTurnSimulationOutput:
    # model best-effort
    if model is None:
        model = (
            getattr(user_profile, "model", None)
            or getattr(the_user, "model", None)
            or getattr(getattr(the_user, "chatbot", None), "model", None)
            or getattr(getattr(the_user, "the_chatbot", None), "model", None)
            or None
        )

    raw_history = getattr(the_user, "conversation_history", None)
    interaction_list = _extract_interaction_list(raw_history)
    paired = _pair_history_as_turns(interaction_list)

    variables_per_turn = getattr(the_user, "variables_per_turn", None)
    if not isinstance(variables_per_turn, list):
        variables_per_turn = []

    retrieved_objs_per_turn = getattr(the_user, "retrieved_objs_per_turn", None)
    if not isinstance(retrieved_objs_per_turn, list):
        retrieved_objs_per_turn = []


    # intents_user_per_turn = getattr(the_user, "intents_user_per_turn", [])
    # intents_system_per_turn = getattr(the_user, "intents_system_per_turn", [])

    # print("intents_user_per_turn", intents_user_per_turn)
    # print("intents_system_per_turn", intents_system_per_turn)

    llm_turns: List[LLMTurn] = []
    for idx, (q, a) in enumerate(paired):
        vars_turn = variables_per_turn[idx] if idx < len(variables_per_turn) else None
        retrieved_turn = retrieved_objs_per_turn[idx] if idx < len(retrieved_objs_per_turn) else None

        content_input = _map_variables_to_content_input(vars_turn)
        content_output_list = _map_retrieved_to_content_output_list(retrieved_turn)

        llm_turns.append(
            LLMTurn(
                question=q,
                answer=a,
                seed=None,
                ordinal_vars=[],
                categorical_vars=[],
                content_input=content_input,              # <-- now populated
                content_output_list=content_output_list,  # <-- now populated
                raw_output=retrieved_turn,                # optional debug
                # question_intent=intents_user_per_turn[idx],
                # answer_intent_classified=intents_system_per_turn[idx],
                poi_exists=None,
                user_intent_influences_fit=None,
            )
        )

    conv = LLMConversation(
        turns=llm_turns,
        seed=None,
        ordinal_vars=[],
        categorical_vars=[],
        continuous_vars=[],
        style_input=None,
        content_input_values={},
        content_input_used=set(),
        # intents_per_turn=intents_user_per_turn
    )

    user_id = getattr(the_user, "user_id", None)
    if user_id is not None:
        conv.assigned_user_id = str(user_id)

    other: Dict[str, Any] = {}
    # try:
    #     other["metadata"] = get_conversation_metadata(user_profile, the_user, serial)
    # except Exception:
    #     other["metadata"] = None

    # other["technology"] = getattr(user_profile, "technology", None)
    # other["test_name"] = getattr(user_profile, "test_name", None)
    other["time_conv"] = time_conv

    return MultiTurnSimulationOutput(
        conversation=conv,
        model=model,
        ipa=ipa or "",
        other=other,
        timestamp=datetime.now(timezone.utc).isoformat(),  # better than serial
    )


# -------------------------
# saving
# -------------------------

def save_simout(simout: Any, eval_result: Dict, out_path: Union[str, os.PathLike], *, pretty: bool = True) -> str:
    if not hasattr(simout, "to_dict") or not callable(getattr(simout, "to_dict")):
        raise TypeError("save_simout expected an object with a callable .to_dict() method")

    out_path = str(out_path)
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    payload = simout.to_dict()

    payload.update(eval_result)

    def _default(o):
        # last-resort safety, mainly for anything odd you put into `other`
        try:
            if hasattr(o, "model_dump"):
                return o.model_dump()
            if hasattr(o, "to_dict"):
                return o.to_dict()
            if is_dataclass(o):
                return o.__dict__
        except Exception:
            pass
        return str(o)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2 if pretty else None,
            default=_default,
        )

    return str(Path(out_path).resolve())