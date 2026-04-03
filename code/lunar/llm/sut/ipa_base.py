import json
import random
import traceback
from abc import abstractmethod
from typing import List, Tuple, Dict, Set, Optional, Any

import numpy as np

from opensbt.simulation.simulator import Simulator
from llm.model.qa_simout import QASimulationOutput
from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.model.models import Utterance, Conversation, Turn
from llm.model.conversation_intents import (
    classify_system_intent,
    UserIntent,
    SYSTEM_TO_USER,
    OPTIMIZABLE_USER_INTENTS,
    select_intent_by_priority,
    PRE_CONFIRMATION_INTENTS,
)
from llm.features import FeatureHandler
from llm.features.models import CombinedFeaturesInstance
from llm.llms import LLMType, pass_llm
from llm.config import LLM_IPA
from llm.prompts import CONVERSATION_FOLLOW_UP_PROMPTS_NAVI

from examples.navi.models import NaviContentInput
from llm.model.models import ContentInput
from examples.navi.navi_utterance_generator import NaviUtteranceGenerator

# Intents that should NOT include content_requirements in their prompts
INTENTS_WITHOUT_CONTENT_REQUIREMENTS = {"ask", "choice", "confirmation", "reject", "stop"}

# Intents that should include ONLY the newly introduced feature
INTENTS_WITH_NEW_FEATURE_ONLY = {"add_preferences", "reject_clarify"}

# Intent that should include ONLY the changed features after resampling
INTENTS_WITH_CHANGED_FEATURES_ONLY = {"change_of_mind"}

# Intent that repeats the previous turn's content features
INTENTS_WITH_REPEAT = {"repeat"}

PRICE_FEATURE_NAME = "fuel_price"


# TODO: NaviContentInput, NaviUtteranceGenerator, CONVERSATION_FOLLOW_UP_PROMPTS_NAVI should be substituted with more generic equivalents
class IPABase(Simulator):
    """
    Abstract base class for all IPAs.

    Provides shared helpers for:
        - content resampling
        - intent priority parsing
        - num_turns extraction
        - content feature initialisation
        - follow-up utterance generation
        - next-intent determination
        - change-of-mind handling
        - repeat handling
    """
    ipa_name: str = "base"
    global_user_counter: int = 0

    @staticmethod
    @abstractmethod
    def simulate(
        list_individuals: List[List[Utterance]],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        max_retries: int = 3,
    ) -> List[QASimulationOutput]:
        """
        Simulate single-turn utterances.
        """
        pass

    @staticmethod
    @abstractmethod
    def simulate_turn(
        user_text: str,
        user_intent: str,
        user_id: str,
        current_content_input: Optional[ContentInput],
        history: List[str],
        max_retries: int = 3,
        llm_type: Optional[str] = None,
        **kwargs,
    ) -> Turn:
        """
        Process a single conversation turn: send the user text, receive the
        system response, and return a Turn object.
        """
        pass

    @staticmethod
    @abstractmethod
    def simulate_conversation(
        list_individuals: List[List[Conversation]],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        config_path: str = "configs/features_simple_judge_navi.json",
        max_retries: int = 3,
        min_turns: int = 2,
        max_turns: int = 5,
    ) -> List[MultiTurnSimulationOutput]:
        """
        Simulate multi-turn conversations.
        """
        pass

    @staticmethod
    def resample_content(
        feature_handler: FeatureHandler,
        original_categorical_vars: List[int],
        original_ordinal_vars: List[float],
        previous_content_input: Optional[ContentInput] = None,
        max_attempts: int = 20,
    ) -> Tuple[List[int], List[float]]:
        """
        Resample categorical variables while preserving
        category and perturbation features. For ordinal variables,
        only rating is resampled.

        If previous_content_input is provided, guarantees that the resulting
        content (after validation) will have at least one non-category feature
        that differs from the previous content.
        
        Price constraint: fuel_price in resampled content must not exceed
        the previous fuel_price (price can only be lowered).
        """
        prev_dump = (
            previous_content_input.model_dump(exclude_none=True)
            if previous_content_input is not None
            else None
        )
        prev_price = prev_dump.get(PRICE_FEATURE_NAME) if prev_dump else None

        for _ in range(max_attempts):
            scores: CombinedFeaturesInstance = feature_handler.sample_feature_scores()

            new_cat = list(scores.categorical)
            cat_keys = list(feature_handler.categorical_features.keys())

            preserved_cat_features = ["category"]
            preserved_cat_features.extend(
                key for key in cat_keys if "perturbation" in key
            )

            for feature_name in preserved_cat_features:
                if feature_name in cat_keys and original_categorical_vars:
                    idx = cat_keys.index(feature_name)
                    if idx < len(new_cat) and idx < len(original_categorical_vars):
                        new_cat[idx] = original_categorical_vars[idx]

            # ordinal: preserve all except rating
            new_ord = list(original_ordinal_vars)
            ord_keys = list(feature_handler.ordinal_features.keys())

            if "rating" in ord_keys:
                idx = ord_keys.index("rating")
                if idx < len(new_ord) and idx < len(scores.ordinal):
                    new_ord[idx] = scores.ordinal[idx]

            # --- PRICE CONSTRAINT during resampling ---
            # If fuel_price was resampled higher than prev_price, force it lower
            if prev_price is not None and PRICE_FEATURE_NAME in cat_keys:
                price_idx = cat_keys.index(PRICE_FEATURE_NAME)
                if price_idx < len(new_cat):
                    feat = feature_handler.categorical_features[PRICE_FEATURE_NAME]
                    new_price_val = feat.values[new_cat[price_idx]]
                    if new_price_val > prev_price:
                        # Pick an index whose value <= prev_price
                        valid_indices = [
                            i for i, v in enumerate(feat.values)
                            if v < prev_price
                        ]
                        if valid_indices:
                            new_cat[price_idx] = random.choice(valid_indices)
                        # else: already at lowest, the value will be cleaned
                        # up later by handle_change_of_mind
            # --- END PRICE CONSTRAINT ---

            # If no previous content to compare against, accept immediately
            if prev_dump is None:
                return new_cat, new_ord

            # Check that at least one non-category feature differs
            new_vals = feature_handler.get_feature_values_dict(
                ordinal_feature_scores=new_ord,
                categorical_feature_indices=new_cat,
            )
            has_difference = False
            for k, v in new_vals.items():
                if k == "category":
                    continue
                if prev_dump.get(k) != v:
                    has_difference = True
                    break

            if has_difference:
                return new_cat, new_ord

        # Exhausted attempts — force a difference on a random non-category feature
        # by re-sampling just that one feature's variable
        new_vals = feature_handler.get_feature_values_dict(
            ordinal_feature_scores=new_ord,
            categorical_feature_indices=new_cat,
        )
        mutable_keys = [
            k for k in new_vals
            if k != "category" and "perturbation" not in k
        ]

        if mutable_keys:
            force_key = random.choice(mutable_keys)
            # Try to pick a different value for this feature
            if force_key in cat_keys:
                idx = cat_keys.index(force_key)
                feat = feature_handler.categorical_features[force_key]
                possible_indices = list(range(len(feat.values)))
                current_idx = new_cat[idx]
                other_indices = [i for i in possible_indices if i != current_idx]
                if other_indices:
                    new_cat[idx] = random.choice(other_indices)
            elif force_key in ord_keys:
                idx = ord_keys.index(force_key)
                # Shift the ordinal score to produce a different value
                feat = feature_handler.ordinal_features[force_key]
                current_val = new_ord[idx]
                # Try a few random scores until a different mapped value is produced
                for _ in range(50):
                    candidate = random.random()
                    if candidate != current_val:
                        new_ord[idx] = candidate
                        break

        return new_cat, new_ord
    
    @staticmethod
    def resample_content_cc(
        feature_handler: FeatureHandler,
        original_categorical_vars: List[int],
        original_ordinal_vars: List[float],
        previous_content_input: Optional[ContentInput] = None,
        max_attempts: int = 20,
    ) -> Tuple[List[int], List[float]]:
        """
        Resample field values for a vehicle control domain.

        - 'system' and 'system2' must remain static.
        - All ordinal variables remain static.
        - All categorical features containing 'perturbation' remain static.
        - Exactly ONE field belonging to one of the systems must be changed,
        and only if its previous value is not None.
        """

        # Dump del contenido previo
        prev_dump = (
            previous_content_input.model_dump(exclude_none=True)
            if previous_content_input is not None
            else None
        )

        cat_keys = list(feature_handler.categorical_features.keys())
        ord_keys = list(feature_handler.ordinal_features.keys())

        # Fields agrupados por sistemas
        SYSTEM_FIELDS = {
            "system": {
                "windows": {
                    "position",
                    "window_state_target",
                    "window_state_initial",
                },
                "fog_lights": {
                    "fog_light_position",
                    "onoff_state_target",
                    "onoff_state_initial",
                },
                "ambient_lights": {
                    "onoff_state_target",
                    "onoff_state_initial",
                },
                "head_lights": {
                    "onoff_state_target",
                    "onoff_state_initial",
                    "head_lights_mode_target",
                    "head_lights_mode_initial",
                },
                "fan": {
                    "onoff_state_target",
                    "onoff_state_initial",
                },
                "reading_lights": {
                    "position",
                    "onoff_state_target",
                    "onoff_state_initial",
                },
                "climate": {
                    "onoff_state_target",
                    "onoff_state_initial",
                    "climate_temperature_value_target",
                    "climate_temperature_value_initial",
                },
                "seat_heating": {
                    "onoff_state_target",
                    "onoff_state_initial",
                    "seat_heating_level_target",
                    "seat_heating_level_initial",
                    "seat_position",
                },
            },
            "system2": {
                "windows2": {
                    "position2",
                    "window_state_target2",
                    "window_state_initial2",
                },
                "fog_lights2": {
                    "fog_light_position2",
                    "onoff_state_target2",
                    "onoff_state_initial2",
                },
                "ambient_lights2": {
                    "onoff_state_target2",
                    "onoff_state_initial2",
                },
                "head_lights2": {
                    "onoff_state_target2",
                    "onoff_state_initial2",
                    "head_lights_mode_target2",
                    "head_lights_mode_initial2",
                },
                "fan2": {
                    "onoff_state_target2",
                    "onoff_state_initial2",
                },
                "reading_lights2": {
                    "position2",
                    "onoff_state_target2",
                    "onoff_state_initial2",
                },
                "climate2": {
                    "onoff_state_target2",
                    "onoff_state_initial2",
                    "climate_temperature_value_target2",
                    "climate_temperature_value_initial2",
                },
                "seat_heating2": {
                    "onoff_state_target2",
                    "onoff_state_initial2",
                    "seat_heating_level_target2",
                    "seat_heating_level_initial2",
                    "seat_position2",
                },
            }
        }

        # Aplanar fields en una lista
        ALL_SYSTEM_FIELDS = {
            field for group in SYSTEM_FIELDS.values() for subsystem in group.values() for field in subsystem
        }

        for _ in range(max_attempts):
            # Sample fresh values
            scores = feature_handler.sample_feature_scores()
            new_cat = list(scores.categorical)
            new_ord = list(original_ordinal_vars)

            for feature_name in ["system", "system2"]:
                if feature_name in cat_keys:
                    idx = cat_keys.index(feature_name)
                    new_cat[idx] = original_categorical_vars[idx]

            for key in cat_keys:
                if "perturbation" in key:
                    idx = cat_keys.index(key)
                    new_cat[idx] = original_categorical_vars[idx]

            new_ord = list(original_ordinal_vars)

            if prev_dump is None:
                return new_cat, new_ord

            new_vals = feature_handler.get_feature_values_dict(
                ordinal_feature_scores=new_ord,
                categorical_feature_indices=new_cat,
            )

            changed_fields = []
            for field in ALL_SYSTEM_FIELDS:
                if field in prev_dump and prev_dump.get(field) is not None:
                    if prev_dump.get(field) != new_vals.get(field):
                        changed_fields.append(field)

            if len(changed_fields) == 1:
                return new_cat, new_ord

        mutable_fields = [
            field for field in ALL_SYSTEM_FIELDS
            if prev_dump.get(field) is not None
        ]

        if mutable_fields:
            forced = random.choice(mutable_fields)

            if forced in cat_keys:
                idx = cat_keys.index(forced)
                feat = feature_handler.categorical_features[forced]
                current_idx = new_cat[idx]
                options = [i for i in range(len(feat.values)) if i != current_idx]
                if options:
                    new_cat[idx] = random.choice(options)

            elif forced in ord_keys:
                idx = ord_keys.index(forced)
                current_val = new_ord[idx]
                for _ in range(50):
                    candidate = random.random()
                    if candidate != current_val:
                        new_ord[idx] = candidate
                        break

        return new_cat, new_ord


    @staticmethod
    def parse_intent_priorities(
        continuous_vars: List[float],
    ) -> Dict[str, float]:
        """
        Map positional continuous variables to named intent priorities.
        """
        if not continuous_vars:
            return {}
        return dict(zip(OPTIMIZABLE_USER_INTENTS, continuous_vars))

    @staticmethod
    def get_num_turns(
        categorical_vars: List[int],
        feature_handler: FeatureHandler,
    ) -> int:
        """
        Extract the number of turns from categorical variables.
        """
        try:
            num_turns_feature = feature_handler.categorical_features["num_turns"]
            num_turns_idx = categorical_vars[-1]
            return num_turns_feature.values[num_turns_idx]
        except Exception:
            return 1

    @staticmethod
    def initialize_content_features(
        initial_content_input: ContentInput,
    ) -> Tuple[Set[str], Set[str], ContentInput]:
        """
        For the first turn, include category and at most 1 other content feature.
        The remaining features are left for future turns.

        Returns:
            all_content_features: all non-None features (excluding category)
            used_content_features: features used in the first turn (excluding category)
            content_input_turn1: NaviContentInput for the first turn
        """
        all_content_features = {
            k
            for k, v in initial_content_input.model_dump().items()
            if v is not None and k != "category"  # don't remove category - not realistic otherwise
        }

        content_input_turn1 = initial_content_input.model_copy()

        # Pick at most 1 feature to include in the first turn
        valid_fields = list(all_content_features)

        if valid_fields:
            # Pick exactly 1 feature to keep; remove the rest
            feature_to_keep = random.choice(valid_fields)
            features_to_remove = [f for f in valid_fields if f != feature_to_keep]

            for field_name in features_to_remove:
                setattr(content_input_turn1, field_name, None)

            used_content_features = {feature_to_keep}
        else:
            used_content_features = set()

        return all_content_features, used_content_features, content_input_turn1

    @staticmethod
    def determine_next_user_intent(
        turn_idx: int,
        min_turns: int,
        max_turns: int,
        processed_turns: List[Turn],
        conversation: Conversation,
        intent_priorities: Dict[str, float],
        unused_content_features: Set[str],
        llm_type: LLMType,
        max_retries: int = 3,
        allow_repeat_intent: bool = True,
        pre_confirmation_intents_used: Optional[Set[str]] = None,
        confirmed: bool = False,
    ) -> Tuple[str, str]:
        """
        Classify the previous system intent and select the next user intent.
        """
        conversation.turns = processed_turns

        try:
            sys_intent = classify_system_intent(conversation, llm_type)
        except Exception as e:
            print(f"[IPABase] classify_system_intent Exception: {e}")
            sys_intent = "misc"

        processed_turns[-1].answer_intent_classified = sys_intent
        prev_user_intent = processed_turns[-1].question_intent

        if sys_intent not in SYSTEM_TO_USER:
            print(f"[IPABase] System intent '{sys_intent}' not in mapping, defaulting to 'misc'")
            sys_intent = "misc"

        possible_user_intents = SYSTEM_TO_USER.get(
            sys_intent, SYSTEM_TO_USER["misc"]
        )

        # determine whether STOP is allowed based on min_turns
        # stop can only appear once at least min_turns have been completed
        allow_stop = turn_idx >= min_turns

        user_intent = select_intent_by_priority(
            possible_intents=possible_user_intents,
            intent_priorities=intent_priorities,
            prev_user_intent=prev_user_intent,
            allow_repeat=(sys_intent in ["reject", "reject_and_followup"]),
            unused_content_features=unused_content_features,
            allow_stop=allow_stop,
            allow_repeat_intent=allow_repeat_intent,
            pre_confirmation_intents_used=pre_confirmation_intents_used,
            confirmed=confirmed,
        )

        return user_intent, sys_intent

    @staticmethod
    def handle_change_of_mind(
        feature_handler: FeatureHandler,
        utterance_gen: NaviUtteranceGenerator,
        categorical_vars: List[int],
        ordinal_vars: List[float],
        previous_content_input: NaviContentInput,
        max_resample_attempts: int = 10,
    ) -> Tuple[NaviContentInput, Dict[str, Any], Set[str], Set[str], Dict[str, Any]]:
        """
        Resample content features for a change-of-mind intent.
        Category is preserved. Other features may change or be omitted.
        Guarantees at least one non-category feature is changed and present
        (not None) in the changed_features dict for the follow-up prompt.

        Returns:
            current_content_input: new NaviContentInput after resampling
            new_vals: full feature values dict
            all_content_features: all non-None features (excluding category) in the new input
            used_content_features: empty set (fresh start for tracking)
            changed_features: dict of features that actually changed (for prompt generation)
        """
        prev_dump = previous_content_input.model_dump(exclude_none=True)
        prev_price = prev_dump.get(PRICE_FEATURE_NAME)

        for attempt in range(max_resample_attempts):
            new_cat_vars, new_ord_vars = IPABase.resample_content(
                feature_handler,
                categorical_vars,
                ordinal_vars,
                previous_content_input=previous_content_input,
            )

            new_vals = feature_handler.get_feature_values_dict(
                ordinal_feature_scores=new_ord_vars,
                categorical_feature_indices=new_cat_vars,
            )

            current_content_input = NaviContentInput.model_validate(new_vals)
            current_content_input = utterance_gen.apply_constraints(current_content_input)

            # price can only be lowered ---
            new_price = getattr(current_content_input, PRICE_FEATURE_NAME, None)
            if new_price is not None and prev_price is not None:
                if new_price >= prev_price:
                    # Try to pick a lower price
                    lower = _pick_lower_price(prev_price, feature_handler)
                    if lower is not None:
                        setattr(current_content_input, PRICE_FEATURE_NAME, lower)
                    else:
                        # Already at lowest — remove price from this change-of-mind
                        setattr(current_content_input, PRICE_FEATURE_NAME, None)

            new_dump = current_content_input.model_dump(exclude_none=True)

            # Compute what actually changed
            changed_features: Dict[str, Any] = {}

            # Features that are new or have a different value
            for k, v in new_dump.items():
                if k == "category":
                    continue
                prev_v = prev_dump.get(k)
                if prev_v != v:
                    changed_features[k] = v

            # Features that were present before but are now absent
            for k in prev_dump:
                if k == "category":
                    continue
                if k not in new_dump:
                    changed_features[k] = None

            # Randomly omit some non-changed features to add variety
            non_changed_keys = [
                k for k, v in new_dump.items()
                if k != "category" and k not in changed_features and v is not None
            ]
            if non_changed_keys:
                num_to_omit = random.randint(0, len(non_changed_keys))
                keys_to_omit = random.sample(non_changed_keys, num_to_omit)
                for k in keys_to_omit:
                    setattr(current_content_input, k, None)
                    changed_features[k] = None

            # Check guarantee: at least one non-category feature with a concrete
            # (non-None) value must be in changed_features
            concrete_changed = {k: v for k, v in changed_features.items() if v is not None}

            if concrete_changed:
                # Guarantee met
                break

            # If no concrete change yet, try to force one:
            # Pick a non-category feature from new_dump that has a value and
            # force it to differ from prev_dump by re-picking
            forceable = [
                k for k, v in new_dump.items()
                if k != "category" and v is not None
            ]

            if forceable:
                # On last attempt, just pick any feature and mark it as changed
                # even if value is same — the LLM will still mention it
                force_key = random.choice(forceable)
                changed_features[force_key] = new_dump[force_key]
                break
            # else: no non-category features at all after constraints, retry

        # If we exhausted attempts and still have no concrete change,
        # fall back: include all non-category non-None features as "changed"
        concrete_changed = {k: v for k, v in changed_features.items() if v is not None}
        if not concrete_changed:
            new_dump_final = current_content_input.model_dump(exclude_none=True)
            for k, v in new_dump_final.items():
                if k != "category":
                    changed_features[k] = v
            # If still nothing (category-only input), that's the edge case
            # where the category has no possible features (e.g., hospital after constraints)

        all_content_features = {
            k
            for k, v in current_content_input.model_dump().items()
            if v is not None and k != "category"
        }
        used_content_features: Set[str] = set()

        return current_content_input, new_vals, all_content_features, used_content_features, changed_features

    @staticmethod
    def handle_repeat(
        previous_turn: Turn,
        used_content_features: Set[str],
    ) -> Tuple[Optional[NaviContentInput], Set[str]]:
        """
        Handle a repeat intent by preserving the same content features
        as in the previous turn.
        
        Returns:
            repeat_content_input: the content input from the previous turn (to reuse)
            repeat_used_features: the same used_content_features (unchanged)
        """
        # Reuse the previous turn's content_input exactly
        if previous_turn.content_input is not None:
            repeat_content_input = previous_turn.content_input.model_copy()
        else:
            repeat_content_input = None

        # Keep the same used_content_features — no new features introduced
        return repeat_content_input, used_content_features

    @staticmethod
    def build_content_requirements_for_intent(
        user_intent: str,
        current_content_input: NaviContentInput,
        used_content_features: Set[str],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build content requirements string for the follow-up prompt based on intent type.

        Rules:
        - ask, choice, confirmation, reject, stop: empty content requirements
        - add_preferences, reject_clarify: only the newly introduced feature
        - change_of_mind: only the changed features after resampling
        """
        # - start (first turn): features selected for turn 1
        

        if user_intent in INTENTS_WITHOUT_CONTENT_REQUIREMENTS:
            return "No specific content requirements for this turn."

        if user_intent in INTENTS_WITH_NEW_FEATURE_ONLY:
            if new_feature_this_turn is not None:
                val = getattr(current_content_input, new_feature_this_turn, None)
                if val is not None:
                    return f"- {new_feature_this_turn}: {val}\n- You MUST reflect this attribute in your turn."
            return "No specific content requirements for this turn."

        if user_intent in INTENTS_WITH_CHANGED_FEATURES_ONLY:
            if changed_features:
                lines = []
                for k, v in changed_features.items():
                    if v is not None:
                        lines.append(f"- {k}: {v}")
                    else:
                        lines.append(f"- {k}: removed/no longer required")
                if lines:
                    lines.append("- You MUST reflect the changed attributes in your turn.")
                    return "\n".join(lines)
            return "No specific content requirements for this turn."

        if user_intent in INTENTS_WITH_REPEAT:
            # For repeat: show all currently used features (same as previous turn)
            d = current_content_input.model_dump(exclude_none=True) if current_content_input else {}
            relevant = {k: v for k, v in d.items() if k == "category" or k in used_content_features}
            if relevant:
                lines = [f"- {k}: {v}" for k, v in relevant.items()]
                lines.append("- You are REPEATING your previous request. Rephrase it but keep the same requirements.")
                return "\n".join(lines)
            return "Repeat your previous request with the same requirements."

        # Default: show all used features up to this point (for start intent etc.)
        d = current_content_input.model_dump(exclude_none=True)
        relevant = {k: v for k, v in d.items() if k == "category" or k in used_content_features}
        if relevant:
            lines = [f"- {k}: {v}" for k, v in relevant.items()]
            lines.append("- You MUST reflect every listed attribute in your turn.")
            return "\n".join(lines)
        return "No specific content requirements for this turn."

    @staticmethod
    def _build_fallback_user_utterance(
        current_content_input: Optional[NaviContentInput],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a manual fallback user utterance starting with 'Find me'
        that includes the relevant content features for this turn.
        """
        parts = ["Find me"]
        feature_parts = []

        # Collect features to mention
        if current_content_input is not None:
            d = current_content_input.model_dump(exclude_none=True)

            # If there's a specific new feature this turn, prioritize it
            if new_feature_this_turn and new_feature_this_turn in d:
                feature_parts.append(f"{new_feature_this_turn}: {d[new_feature_this_turn]}")
            elif changed_features:
                for k, v in changed_features.items():
                    if v is not None:
                        feature_parts.append(f"{k}: {v}")

            else:
                # Include category at minimum
                if "category" in d:
                    feature_parts.append(str(d["category"]).replace("_", " "))

        if feature_parts:
            parts.append("a place with " + ", ".join(feature_parts))
        else:
            parts.append("a suitable place nearby")

        return " ".join(parts)

    @staticmethod
    def generate_follow_up_utterance(
        user_intent: str,
        feature_values: Dict[str, Any],
        current_content_input: NaviContentInput,
        history: List[str],
        utterance_gen: NaviUtteranceGenerator,
        llm_type: LLMType,
        used_content_features: Set[str],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Generate a follow-up user utterance for a given intent.
        """
        prompt_template = CONVERSATION_FOLLOW_UP_PROMPTS_NAVI.get(
            user_intent,
            CONVERSATION_FOLLOW_UP_PROMPTS_NAVI.get("repeat",
                CONVERSATION_FOLLOW_UP_PROMPTS_NAVI["ask"]),
        )

        style_prompt_str = utterance_gen._style_prompt(feature_values)

        content_req_str = IPABase.build_content_requirements_for_intent(
            user_intent=user_intent,
            current_content_input=current_content_input,
            used_content_features=used_content_features,
            new_feature_this_turn=new_feature_this_turn,
            changed_features=changed_features,
        )

        dialogue_history_str = "\n".join(history)

        full_prompt = prompt_template.format(
            history=dialogue_history_str,
            style_prompt=style_prompt_str,
            content_requirements=content_req_str,
        )
        # print(f"[IPABase] Full prompt for follow-up generation:\n{full_prompt}\n")

        user_text = None
        for attempt in range(max_retries):
            try:
                user_text = pass_llm(full_prompt, llm_type=llm_type, temperature=0.7)
                if user_text is not None and user_text.strip() != "":
                    break
                else:
                    user_text = None
                    print(f"[IPABase] generate_follow_up_utterance attempt {attempt + 1} returned empty string")
            except Exception:
                print(f"[IPABase] generate_follow_up_utterance attempt {attempt + 1} failed:")
                traceback.print_exc()

        # Fallback: build a manual "Find me ..." utterance with relevant features
        if user_text is None or user_text.strip() == "":
            user_text = IPABase._build_fallback_user_utterance(
                current_content_input=current_content_input,
                new_feature_this_turn=new_feature_this_turn,
                changed_features=changed_features,
            )
            print(f"[IPABase] Using fallback user utterance: {user_text}")

        return utterance_gen._apply_post_perturbations(user_text, feature_values)

    @staticmethod
    def generate_first_utterance(
        conversation: Conversation,
        utterance_gen: NaviUtteranceGenerator,
        content_input_turn1: NaviContentInput,
        llm_type: LLMType,
        max_retries: int = 3,
    ) -> Utterance:
        """
        Generate the first user utterance for a conversation.
        Retries on failure/empty and falls back to a manual 'Find me ...' utterance.
        """
        last_exception = None
        for attempt in range(max_retries):
            try:
                utterance = utterance_gen.generate_utterance(
                    seed=conversation.seed,
                    ordinal_vars=conversation.ordinal_vars,
                    categorical_vars=conversation.categorical_vars,
                    llm_type=llm_type,
                    content_input_override=content_input_turn1,
                )
                if utterance.question is not None and utterance.question.strip() != "":
                    return utterance
                else:
                    print(f"[IPABase] generate_first_utterance attempt {attempt + 1} returned empty question")
            except Exception as e:
                print(f"[IPABase] generate_first_utterance attempt {attempt + 1} failed: {e}")
                traceback.print_exc()
                last_exception = e

        # Fallback
        fallback_question = IPABase._build_fallback_user_utterance(
            current_content_input=content_input_turn1,
        )
        print(f"[IPABase] Using fallback first utterance: {fallback_question}")
        return Utterance(
            question=fallback_question,
            seed=conversation.seed,
            ordinal_vars=conversation.ordinal_vars,
            categorical_vars=conversation.categorical_vars,
            content_input=content_input_turn1,
        )

    @staticmethod
    def prepare_conversation_state(
        conversation: Conversation,
        feature_handler: FeatureHandler,
        utterance_gen: NaviUtteranceGenerator,
        min_turns: int = 2,
        max_turns: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute all derived state needed before the conversation loop.
        """
        intent_priorities = IPABase.parse_intent_priorities(
            conversation.continuous_vars
        )
        feature_values = feature_handler.get_feature_values_dict(
            ordinal_feature_scores=conversation.ordinal_vars,
            categorical_feature_indices=conversation.categorical_vars,
        )
        initial_content_input = NaviContentInput.model_validate(feature_values)
        initial_content_input = utterance_gen.apply_constraints(initial_content_input)

        all_cf, used_cf, content_input_turn1 = IPABase.initialize_content_features(
            initial_content_input
        )

        return {
            "min_turns": min_turns,
            "max_turns": max_turns,
            "intent_priorities": intent_priorities,
            "feature_values": feature_values,
            "initial_content_input": initial_content_input,
            "all_content_features": all_cf,
            "used_content_features": used_cf,
            "content_input_turn1": content_input_turn1,
        }

    @classmethod
    def run_conversation_loop(
        cls,
        conversation: Conversation,
        feature_handler: FeatureHandler,
        utterance_gen: NaviUtteranceGenerator,
        llm_type: LLMType,
        context: object = None,
        max_retries: int = 3,
        min_turns: int = 2,
        max_turns: int = 5,
        max_repeats: int = 2,
        llm_type_utterance_gen: Optional[LLMType] = None,
        **turn_kwargs,
    ) -> Conversation:
        """
        Template-method that drives the multi-turn conversation loop.
        
        Args:
            max_repeats: Maximum number of repeat intents allowed in a conversation.
        """
        if llm_type_utterance_gen is None:
            llm_type_utterance_gen = llm_type

        state = cls.prepare_conversation_state(
            conversation, feature_handler, utterance_gen,
            min_turns=min_turns, max_turns=max_turns,
        )

        min_turns = state["min_turns"]
        max_turns = state["max_turns"]
        intent_priorities = state["intent_priorities"]
        feature_values = state["feature_values"]
        initial_content_input = state["initial_content_input"]
        all_content_features = state["all_content_features"]
        used_content_features = state["used_content_features"]
        content_input_turn1 = state["content_input_turn1"]

        # first utterance — use utterance generation LLM
        utterance_obj = cls.generate_first_utterance(
            conversation, utterance_gen, content_input_turn1, llm_type_utterance_gen
        )

        user_text = utterance_obj.question
        user_intent = UserIntent.START.value
        current_content_input = initial_content_input

        processed_turns: List[Turn] = []
        history: List[str] = []

        # Track per-turn state
        new_feature_this_turn: Optional[str] = None
        changed_features: Optional[Dict[str, Any]] = None

        # Track repeat count
        repeat_count: int = 0

        # Track which pre-confirmation intents have been used (need >= 2 distinct for confirmation)
        pre_confirmation_intents_used: Set[str] = set()

        # Track whether confirmation has been used (blocks all pre-confirmation intents after)
        confirmed: bool = False

        for turn_idx in range(max_turns):
            # follow-up generation (not for the first turn)
            if turn_idx > 0:
                unused_content_features = all_content_features - used_content_features

                # Check whether repeat intent is still allowed
                allow_repeat_intent = repeat_count < max_repeats

                # Intent classification uses llm_type (LLM_IPA)
                user_intent, sys_intent = cls.determine_next_user_intent(
                    turn_idx=turn_idx,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    processed_turns=processed_turns,
                    conversation=conversation,
                    intent_priorities=intent_priorities,
                    unused_content_features=unused_content_features,
                    llm_type=llm_type,
                    max_retries=max_retries,
                    allow_repeat_intent=allow_repeat_intent,
                    pre_confirmation_intents_used=pre_confirmation_intents_used,
                    confirmed=confirmed,
                )

                # Reset per-turn trackers
                new_feature_this_turn = None
                changed_features = None

                # Update tracking based on selected intent
                if user_intent == UserIntent.CONFIRMATION.value:
                    confirmed = True

                # Track pre-confirmation intents usage
                if user_intent in PRE_CONFIRMATION_INTENTS:
                    pre_confirmation_intents_used.add(user_intent)

                # handle repeat intent
                if user_intent == UserIntent.REPEAT.value:
                    repeat_count += 1
                    previous_turn = processed_turns[-1]
                    repeat_content_input, used_content_features = cls.handle_repeat(
                        previous_turn=previous_turn,
                        used_content_features=used_content_features,
                    )
                    if repeat_content_input is not None:
                        current_content_input = repeat_content_input

                # handle change of mind
                elif user_intent == UserIntent.CHANGE_OF_MIND.value:
                    (
                        current_content_input,
                        feature_values,
                        all_content_features,
                        used_content_features,
                        changed_features,
                    ) = cls.handle_change_of_mind(
                        feature_handler=feature_handler,
                        utterance_gen=utterance_gen,
                        categorical_vars=conversation.categorical_vars,
                        ordinal_vars=conversation.ordinal_vars,
                        previous_content_input=current_content_input,
                    )

                # handle add_preferences / reject_clarify
                elif user_intent in [
                    UserIntent.ADD_PREFERENCES.value,
                    UserIntent.REJECT_CLARIFY.value,
                ]:
                    if type(utterance_gen) == NaviUtteranceGenerator:
                        unused = all_content_features - used_content_features
                        if unused:
                            # Price constraint for add_preferences / reject_clarify
                            # If fuel_price is among unused candidates, check whether
                            # lowering is possible.  If not, exclude it from candidates.
                            prev_price = getattr(current_content_input, PRICE_FEATURE_NAME, None)
                            filtered_unused = set(unused)

                            if PRICE_FEATURE_NAME in filtered_unused and prev_price is not None:
                                if _is_lowest_price(prev_price, feature_handler):
                                    # Can't lower further — remove fuel_price as candidate
                                    filtered_unused.discard(PRICE_FEATURE_NAME)

                            if filtered_unused:
                                feature_to_add = random.choice(list(filtered_unused))

                                # If fuel_price was selected, replace its value with a lower one
                                if feature_to_add == PRICE_FEATURE_NAME and prev_price is not None:
                                    lower = _pick_lower_price(prev_price, feature_handler)
                                    if lower is not None:
                                        setattr(current_content_input, PRICE_FEATURE_NAME, lower)
                                    else:
                                        # Shouldn't happen (we filtered above), but be safe
                                        filtered_unused.discard(PRICE_FEATURE_NAME)
                                        if filtered_unused:
                                            feature_to_add = random.choice(list(filtered_unused))
                                        else:
                                            feature_to_add = None

                                if feature_to_add is not None:
                                    used_content_features.add(feature_to_add)
                                    new_feature_this_turn = feature_to_add
                            # If filtered_unused is empty, no feature can be added
                            # — the intent will proceed without introducing new content
                    else:
                        unused = all_content_features - used_content_features
                        allowed_fields_by_system = {
                            "windows": {
                                "position",
                                "window_state_target",
                                "window_state_initial",
                            },
                            "fog_lights": {
                                "fog_light_position",
                                "onoff_state_target",
                                "onoff_state_initial",
                            },
                            "ambient_lights": {
                                "onoff_state_target",
                                "onoff_state_initial",
                            },
                            "head_lights": {
                                "onoff_state_target",
                                "onoff_state_initial",
                                "head_lights_mode_target",
                                "head_lights_mode_initial",
                            },
                            "fan": {
                                "onoff_state_target",
                                "onoff_state_initial",
                            },
                            "reading_lights": {
                                "position",
                                "onoff_state_target",
                                "onoff_state_initial",
                            },
                            "climate": {
                                "onoff_state_target",
                                "onoff_state_initial",
                                "climate_temperature_value_target",
                                "climate_temperature_value_initial",
                            },
                            "seat_heating": {
                                "onoff_state_target",
                                "onoff_state_initial",
                                "seat_heating_level_target",
                                "seat_heating_level_initial",
                                "seat_position",
                            },
                            "windows2": {
                                "position2",
                                "window_state_target2",
                                "window_state_initial2",
                            },
                            "fog_lights2": {
                                "fog_light_position2",
                                "onoff_state_target2",
                                "onoff_state_initial2",
                            },
                            "ambient_lights2": {
                                "onoff_state_target2",
                                "onoff_state_initial2",
                            },
                            "head_lights2": {
                                "onoff_state_target2",
                                "onoff_state_initial2",
                                "head_lights_mode_target2",
                                "head_lights_mode_initial2",
                            },
                            "fan2": {
                                "onoff_state_target2",
                                "onoff_state_initial2",
                            },
                            "reading_lights2": {
                                "position2",
                                "onoff_state_target2",
                                "onoff_state_initial2",
                            },
                            "climate2": {
                                "onoff_state_target2",
                                "onoff_state_initial2",
                                "climate_temperature_value_target2",
                                "climate_temperature_value_initial2",
                            },
                            "seat_heating2": {
                                "onoff_state_target2",
                                "onoff_state_initial2",
                                "seat_heating_level_target2",
                                "seat_heating_level_initial2",
                                "seat_position2",
                            },
                        }
                        
                        system = getattr(current_content_input, "system", None)
                        system2 = getattr(current_content_input, "system2", None)

                        set1 = allowed_fields_by_system.get(system, set())
                        set2 = allowed_fields_by_system.get(system2, set())

                        available = []
                        for value in unused:
                            if value in set1 or value in set2:
                                available.append(value)
                        if len(available)>0:
                            feature_to_add = random.choice(available)
                            used_content_features.add(feature_to_add)
                            new_feature_this_turn = feature_to_add

                # Follow-up utterance generation uses llm_type_utterance_gen (LLM_TYPE)
                user_text = cls.generate_follow_up_utterance(
                    user_intent=user_intent,
                    feature_values=feature_values,
                    current_content_input=current_content_input,
                    history=history,
                    utterance_gen=utterance_gen,
                    llm_type=llm_type_utterance_gen,
                    used_content_features=used_content_features,
                    new_feature_this_turn=new_feature_this_turn,
                    changed_features=changed_features,
                    max_retries=max_retries,
                )

            # Build content input for this specific turn
            turn_content_input = cls._build_turn_content_input(
                current_content_input=current_content_input,
                used_content_features=used_content_features,
                new_feature_this_turn=new_feature_this_turn,
                user_intent=user_intent,
                content_input_turn1=content_input_turn1 if turn_idx == 0 else None,
            )

            # delegate turn execution to the subclass — with retry and fallback
            turn = None
            for attempt in range(max_retries):
                try:
                    turn = cls.simulate_turn(
                        user_text=user_text,
                        user_intent=user_intent,
                        user_id=str(conversation.assigned_user_id),
                        current_content_input=turn_content_input,
                        history=history,
                        max_retries=max_retries,
                        context=context,
                        conversation=conversation,
                        **turn_kwargs,
                    )
                    if turn is not None and turn.answer is not None and turn.answer.strip() != "":

                        if processed_turns == []:
                            processed_turns = [turn]
                        break
                    else:
                        print(f"[IPABase] simulate_turn attempt {attempt + 1} returned empty system response")
                        turn = None
                except Exception as e:
                    print(f"[IPABase] simulate_turn attempt {attempt + 1} failed: {e}")
                    traceback.print_exc()
                    turn = None

            # Fallback if all retries failed or system response is empty
            if turn is None or turn.answer is None or turn.answer.strip() == "":
                if turn is None:
                    turn = Turn(
                        question=user_text,
                        answer=IPABase.SYSTEM_FALLBACK_RESPONSE,
                        question_intent=user_intent,
                        content_input=turn_content_input.model_copy() if turn_content_input else None,
                        content_output_list=[],
                        poi_exists=False,
                    )
                else:
                    turn.answer = IPABase.SYSTEM_FALLBACK_RESPONSE
                history.append(f"User: {user_text}")
                history.append(f"System: {turn.answer}")
                print(f"[IPABase] Using fallback system response for turn {turn_idx}")

        conversation.turns = processed_turns
        conversation.content_input_used = used_content_features

        return conversation

    @staticmethod
    def _build_turn_content_input(
        current_content_input: NaviContentInput,
        used_content_features: Set[str],
        new_feature_this_turn: Optional[str],
        user_intent: str,
        content_input_turn1: Optional[NaviContentInput] = None,
    ) -> NaviContentInput:
        """
        Build a NaviContentInput for a specific turn that only includes:
        - category (always)
        - features used in previous turns
        - the feature introduced in this turn (if any)

        For the first turn (start intent), use content_input_turn1 directly.
        """
        if user_intent == UserIntent.START.value and content_input_turn1 is not None:
            return content_input_turn1

        # Build from current_content_input, keeping only category + used features
        full_dump = current_content_input.model_dump()
        filtered: Dict[str, Any] = {}

        # Always include category
        filtered["category"] = full_dump.get("category")

        # Include used features
        for feat in used_content_features:
            if feat in full_dump and full_dump[feat] is not None:
                filtered[feat] = full_dump[feat]

        # Include new feature for this turn
        if new_feature_this_turn and new_feature_this_turn in full_dump:
            if full_dump[new_feature_this_turn] is not None:
                filtered[new_feature_this_turn] = full_dump[new_feature_this_turn]

        return NaviContentInput.model_validate(filtered)

IPABase.SYSTEM_FALLBACK_RESPONSE = "Unfortunately, something failed."

def _get_ordered_price_values(feature_handler: FeatureHandler) -> List[float]:
    """Return the fuel_price feature values in ascending order, or [] if not found."""
    feat = feature_handler.categorical_features.get(PRICE_FEATURE_NAME)
    if feat is None:
        feat = feature_handler.ordinal_features.get(PRICE_FEATURE_NAME)
    if feat is None:
        return []
    return sorted(feat.values)


def _is_lowest_price(current_price, feature_handler: FeatureHandler) -> bool:
    """Check whether current_price is already the lowest available value."""
    ordered = _get_ordered_price_values(feature_handler)
    if not ordered or current_price is None:
        return False
    return current_price <= ordered[0]


def _pick_lower_price(current_price, feature_handler: FeatureHandler):
    """Pick a random price strictly lower than current_price, or None if impossible."""
    ordered = _get_ordered_price_values(feature_handler)
    if not ordered or current_price is None:
        return None
    candidates = [p for p in ordered if p < current_price]
    if not candidates:
        return None
    return random.choice(candidates)
