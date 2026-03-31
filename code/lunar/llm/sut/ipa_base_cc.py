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
#from llm.prompts import CONVERSATION_FOLLOW_UP_PROMPTS_NAVI
from llm.prompts import CONVERSATION_FOLLOW_UP_PROMPTS_CAR_CONTROL

from examples.car_control.models_new import CCContentInput
from examples.car_control.cc_utterance_generator_new import CCUtteranceGenerator
from llm.model.models import ContentInput

# Intents that should NOT include content_requirements in their prompts
INTENTS_WITHOUT_CONTENT_REQUIREMENTS = {"ask", "choice", "confirmation", "reject", "stop"}

# Intents that should include ONLY the newly introduced feature
INTENTS_WITH_NEW_FEATURE_ONLY = {"add_preferences", "reject_clarify"}

# Intent that should include ONLY the changed features after resampling
INTENTS_WITH_CHANGED_FEATURES_ONLY = {"change_of_mind"}

# Intent that repeats the previous turn's content features
INTENTS_WITH_REPEAT = {"repeat"}

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
    #@abstractmethod
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
    #@abstractmethod
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
    #@abstractmethod
    def simulate_conversation(
        list_individuals: List[List[Conversation]],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        config_path: str = "configs/features_simple_judge_cc.json",
        max_retries: int = 3,
        min_turns: int = 2,
        max_turns: int = 3,
    ) -> List[MultiTurnSimulationOutput]:
        """
        Simulate multi-turn conversations.
        """
        pass
    
    @staticmethod
    def resample_content(
        feature_handler: "FeatureHandler",
        original_categorical_vars: List[int],
        original_ordinal_vars: List[float],
        previous_content_input: Optional["ContentInput"] = None,
        max_attempts: int = 20,
    ) -> Tuple[List[int], List[float]]:
        """
        Resample field values for a vehicle control domain.
        """
        prev_dump = (
            previous_content_input.model_dump(exclude_none=True)
            if previous_content_input is not None
            else None
        )

        cat_keys = list(feature_handler.categorical_features.keys())
        ord_keys = list(feature_handler.ordinal_features.keys())

        locked_categorical = set()

        #for fname in ("system", "system2"):
        #    if fname in cat_keys:
        #        locked_categorical.add(fname)

        for key in cat_keys:
            if "perturbation" in key.lower():
                locked_categorical.add(key)

        system_initial_keys = set()
        for group in SYSTEM_FIELDS.values():
            for subsystem_fields in group.values():
                for field in subsystem_fields:
                    if "initial" in field.lower() and field in cat_keys:
                        system_initial_keys.add(field)
        locked_categorical.update(system_initial_keys)

        for _ in range(max_attempts):
            scores = feature_handler.sample_feature_scores()
            new_cat = list(scores.categorical)

            new_ord = list(original_ordinal_vars)

            for key in locked_categorical:
                idx = cat_keys.index(key)
                new_cat[idx] = original_categorical_vars[idx]

            changed = any(
                (new_cat[i] != original_categorical_vars[i])
                for i, key in enumerate(cat_keys)
                if key not in locked_categorical
            )
            if changed:
                return new_cat, new_ord

        allowed_indices = [
            i for i, key in enumerate(cat_keys)
            if key not in locked_categorical
        ]

        if allowed_indices:
            new_cat = list(original_categorical_vars)
            i = random.choice(allowed_indices)
            feat = feature_handler.categorical_features[cat_keys[i]]
            current_idx = original_categorical_vars[i]
            options = [j for j in range(len(feat.values)) if j != current_idx]
            if options:
                new_cat[i] = random.choice(options)
            new_ord = list(original_ordinal_vars)
            return new_cat, new_ord

        return list(original_categorical_vars), list(original_ordinal_vars)


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
        Inicialización para el dominio de control del coche.

        - No existe 'category' en este dominio.
        - 'system' y 'system2' cumplen el rol de fields centrales.
        - No se elimina ninguna feature en el primer turno.
        """

        # Todas las features no vacías
        all_content_features = {
            k
            for k, v in initial_content_input.model_dump().items()
            if v is not None
        }

        # En el dominio del coche queremos usar todas desde el inicio
        used_content_features = set(all_content_features)

        # Simplemente devolvemos la copia íntegra
        content_input_turn1 = initial_content_input.model_copy()

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
        utterance_gen: CCUtteranceGenerator,
        categorical_vars: List[int],
        ordinal_vars: List[float],
        previous_content_input: CCContentInput,
        max_resample_attempts: int = 10,
    ) -> Tuple[CCContentInput, Dict[str, Any], Set[str], Set[str], Dict[str, Any]]:

        # log inputs 
        print("previous_content_input:", previous_content_input)


        prev_dump = previous_content_input.model_dump(exclude_none=True)
        prev_system = prev_dump["system"]
        prev_perturbation = prev_dump.get("word_perturbation")
        prev_politeness = prev_dump["politeness"]
        prev_anthr = prev_dump["anthropomorphism"]
        prev_slang = prev_dump["slang"]
        prev_implicitness = prev_dump["implicitness"]

        # Campos permitidos del sistema original
        allowed_fields = SYSTEM_FIELDS["system"][prev_system]

        changed_features: Dict[str, Any] = {}

        for attempt in range(max_resample_attempts):

            # ▼ 1. Resampling normal
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

            current_content_input = CCContentInput.model_validate(new_vals)
            current_content_input = utterance_gen.apply_constraints(current_content_input)

            new_dump = current_content_input.model_dump(exclude_none=True)

            # ▼ 2. System never changes
            new_dump["system"] = prev_system

            # ▼ 3. Detect available fields
            changed_features = {}
            changed_allowed_fields = []

            for k in allowed_fields:
                prev_v = prev_dump.get(k)
                new_v = new_dump.get(k)

                # A change is necessary
                if new_v is not None and prev_v != new_v:
                    changed_features[k] = new_v
                    changed_allowed_fields.append(k)

            # ▼ 4. Accept resampling if one field changed
            if changed_allowed_fields:
                break

        # ▼ 5. Rebuild input
        new_dump["system"] = prev_system
        if prev_perturbation:
            new_dump["word_perturbation"] = prev_perturbation
        new_dump["politeness"] = prev_politeness
        new_dump["anthropomorphism"] = prev_anthr
        new_dump["slang"] = prev_slang
        new_dump["implicitness"] = prev_implicitness
        current_content_input = CCContentInput.model_validate(new_dump)

        all_content_features = {
            k for k, v in current_content_input.model_dump().items() if v is not None
        }

        used_content_features: Set[str] = set()
        print("changed_features:", changed_features)
        print("new_dump:", new_dump)
        print("current_content_input:", current_content_input)
                
        return (
            current_content_input,
            new_dump,
            all_content_features,
            used_content_features,
            changed_features,
        )



    @staticmethod
    def handle_repeat(
        previous_turn: Turn,
        used_content_features: Set[str],
    ) -> Tuple[Optional[CCContentInput], Set[str]]:
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
        current_content_input: CCContentInput,
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
                    return f"- System: {current_content_input.system}\n- {new_feature_this_turn}: {val}\n- You MUST reflect this attribute in your turn."
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
            relevant = {k: v for k, v in d.items() if k == "system" or k == "system2" or k in used_content_features}
            if relevant:
                lines = [f"- {k}: {v}" for k, v in relevant.items()]
                lines.append("- You are REPEATING your previous request. Rephrase it but keep the same requirements.")
                return "\n".join(lines)
            return "Repeat your previous request with the same requirements."

        # Default: show all used features up to this point (for start intent etc.)
        d = current_content_input.model_dump(exclude_none=True)
        relevant = {k: v for k, v in d.items() if k == "system" or k == "system2" or k in used_content_features}
        if relevant:
            lines = [f"- {k}: {v}" for k, v in relevant.items()]
            lines.append("- You MUST reflect every listed attribute in your turn.")
            return "\n".join(lines)
        return "No specific content requirements for this turn."
    
    @staticmethod
    def _build_fallback_user_utterance(
        current_content_input: Optional[CCContentInput],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a fallback user utterance for the car control domain.
        The fallback describes only the real changes detected in changed_features,
        with natural English phrasing.
        """

        # If no input available, return a generic request
        if current_content_input is None:
            return "I want to adjust something in the car."

        # Priority 1: if there is a specific feature highlighted this turn
        if new_feature_this_turn and changed_features:
            if new_feature_this_turn in changed_features:
                val = changed_features[new_feature_this_turn]
                return f"Now I want {new_feature_this_turn.replace('_', ' ')} to be {val}."

        # Priority 2: describe all real changes
        if changed_features:
            parts = []
            for k, v in changed_features.items():
                if v is not None:
                    #parts.append(f"{k.replace('_', ' ')} set to {v}")
                    parts.append(f"Set it to {v}")
            if parts:
                #return "Now I want to change " + ", ".join(parts) + "."
                return ". ".join(parts)

        # Priority 3: fully generic fallback
        return "Which settings can I change?"



    @staticmethod
    def generate_follow_up_utterance(
        user_intent: str,
        feature_values: Dict[str, Any],
        current_content_input: CCContentInput,
        history: List[str],
        utterance_gen: CCUtteranceGenerator,
        llm_type: LLMType,
        used_content_features: Set[str],
        new_feature_this_turn: Optional[str] = None,
        changed_features: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Generate a follow-up user utterance for a given intent.
        """
        prompt_template = CONVERSATION_FOLLOW_UP_PROMPTS_CAR_CONTROL.get(
            user_intent,
            CONVERSATION_FOLLOW_UP_PROMPTS_CAR_CONTROL.get("repeat",
                CONVERSATION_FOLLOW_UP_PROMPTS_CAR_CONTROL["ask"]),
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
        #print("###################################")
        #print("Information")
        #print("")
        #print(f"Intent: {user_intent}")
        #print("")
        #print(f"History:\n{dialogue_history_str}")
        #print("")
        #print(f"Requirements:\n{content_req_str}")
        #print("###################################")
        # print(f"[IPABase] Full prompt for follow-up generation:\n{full_prompt}\n")

        user_text = None
        for attempt in range(max_retries):
            try:
                #print("")
                #print("##################################")
                #print(full_prompt)
                #print("##################################")
                #print("")
                user_text = pass_llm(full_prompt, llm_type=llm_type, temperature=0.7)
                #print("")
                #print("##################################")
                #print(user_text)
                #print("##################################")
                #print("")
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
        utterance_gen: CCUtteranceGenerator,
        content_input_turn1: CCContentInput,
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
        utterance_gen: CCUtteranceGenerator,
        min_turns: int = 2,
        max_turns: int = 3,
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
        initial_content_input = CCContentInput.model_validate(feature_values)
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
        utterance_gen: CCUtteranceGenerator,
        llm_ipa: LLMType,
        llm_classifier: LLMType,
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
        change_of_mind_intents = False
        if llm_type_utterance_gen is None:
            llm_type_utterance_gen = LLMType.DEEPSEEK_V3_0324

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
                    llm_type=llm_classifier,
                    max_retries=max_retries,
                    allow_repeat_intent=allow_repeat_intent,
                    pre_confirmation_intents_used=pre_confirmation_intents_used,
                    confirmed=confirmed,
                )

                if user_intent == UserIntent.CHANGE_OF_MIND.value and change_of_mind_intents:
                    user_intent = UserIntent.REPEAT.value

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
                    change_of_mind_intents = True
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

                    # Allowed fields
                    allowed_fields_by_system = {
                        "windows": {
                            "position",
                            "window_state_target",
                        },
                        "fog_lights": {
                            "fog_light_position",
                            "onoff_state_target",
                        },
                        "ambient_lights": {
                            "onoff_state_target",
                        },
                        "head_lights": {
                            "onoff_state_target",
                            "head_lights_mode_target",
                        },
                        "fan": {
                            "onoff_state_target",
                        },
                        "reading_lights": {
                            "position",
                            "onoff_state_target",
                        },
                        "climate": {
                            "onoff_state_target",
                            "climate_temperature_value_target",
                        },
                        "seat_heating": {
                            "onoff_state_target",
                            "seat_heating_level_target",
                            "seat_position",
                        }
                    }
                    """
                        # Second system
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
                    """

                    # Active systems
                    system = getattr(current_content_input, "system", None)
                    #system2 = getattr(current_content_input, "system2", None)

                    # Allowed fields combined
                    allowed_union = set()
                    if system in allowed_fields_by_system:
                        allowed_union |= allowed_fields_by_system[system]
                    #if system2 in allowed_fields_by_system:
                    #    allowed_union |= allowed_fields_by_system[system2]

                    # Domain blocks
                    def is_blocked(field_name: str) -> bool:
                        name = field_name.lower()
                        return ("initial" in name) or ("perturbation" in name)

                    dump = current_content_input.model_dump()

                    none_candidates = [
                        f for f in allowed_union
                        if f in dump and dump[f] is None and not is_blocked(f)
                    ]

                    selected_field = None

                    if none_candidates:
                        preferred = [f for f in none_candidates if "target" in f]  # opcional
                        pool = preferred if preferred else none_candidates
                        selected_field = random.choice(pool)

                        cat_keys = feature_handler.categorical_features.keys()
                        ord_keys = feature_handler.ordinal_features.keys()

                        if selected_field in cat_keys:
                            feat = feature_handler.categorical_features[selected_field]
                            new_val = random.choice(list(feat.values))
                            setattr(current_content_input, selected_field, new_val)

                        elif selected_field in ord_keys:
                            setattr(current_content_input, selected_field, 0.5)  # si permites ordinales
                        else:
                            setattr(current_content_input, selected_field, True)

                        used_content_features.add(selected_field)
                        new_feature_this_turn = selected_field

                    else:
                        # Fallback cuando NO hay ningún allowed field con None
                        available = [
                            f for f in allowed_union
                            if f not in used_content_features and not is_blocked(f)
                        ]

                        if available:
                            selected_field = random.choice(available)
                            used_content_features.add(selected_field)
                            new_feature_this_turn = selected_field

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
                        llm_type=llm_ipa,
                        **turn_kwargs,
                    )
                    if turn is not None and turn.answer is not None and turn.answer.strip() != "":
                        if processed_turns == []:
                            processed_turns = [turn]
                        else:
                            processed_turns.append(turn)
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

                processed_turns.append(turn)
                history.append(f"User: {user_text}")
                history.append(f"System: {turn.answer}")
                print(f"[IPABase] Using fallback system response for turn {turn_idx}")
            
            if user_intent == "stop":
                break

        conversation.turns = processed_turns
        conversation.content_input_used = used_content_features

        return conversation

    
    @staticmethod
    def _build_turn_content_input(
        current_content_input: CCContentInput,
        used_content_features: Set[str],
        new_feature_this_turn: Optional[str],
        user_intent: str,
        content_input_turn1: Optional[CCContentInput] = None,
    ) -> CCContentInput:
        """
        Car Control domain:
        - No 'category'.
        - Always preserve full state; do NOT hide features by turn.
        - Ensure 'system' and 'system2' are present if available.
        - For START, return content_input_turn1 if provided (it should already be a full copy).
        """

        # For first turn, keep the initial content as-is
        if user_intent == UserIntent.START.value and content_input_turn1 is not None:
            return content_input_turn1

        # Recommended: return the full, current content as-is (no feature hiding)
        # Option A (strict mirror): return current_content_input directly
        # return current_content_input

        # Option B (normalized copy): include all non-None fields
        full_dump = current_content_input.model_dump()
        filtered: Dict[str, Any] = {
            k: v for k, v in full_dump.items() if ((v is not None and not isinstance(v, bool)) or k=="onoff_state_target")
        }

        # Explicitly include 'system' and 'system2' even if they were None in dump
        # (uncomment if you want to carry them through regardless)
        if "system" in full_dump:
            filtered["system"] = full_dump["system"]
        if "system2" in full_dump:
            filtered["system2"] = full_dump["system2"]

        if isinstance(filtered.get("onoff_state_target"), bool):
            filtered["onoff_state_target"] = "on" if filtered["onoff_state_target"] else "off"
        if isinstance(filtered.get("onoff_state_target"), bool):
            filtered["onoff_state_target2"] = "on" if filtered["onoff_state_target2"] else "off"
        if isinstance(filtered.get("onoff_state_initial"), bool):
            filtered["onoff_state_initial"] = "on" if filtered["onoff_state_initial"] else "off"
        if isinstance(filtered.get("onoff_state_initial"), bool):
            filtered["onoff_state_initial2"] = "on" if filtered["onoff_state_initial2"] else "off"
        return CCContentInput.model_validate(filtered)


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


if __name__ == "__main__":
    print("all correct?")
    print("Has method:", hasattr(IPABase, "run_conversation_loop"))
    print("")
    ipa = IPABase()
    featurehandler = FeatureHandler.from_json("configs/features_simple_judge_cc.json")
    ordinal = [0.4556738770187592, 0.4611940330799759, 0.02026152013051903, 0.804308792517508]
    categorical = [0, 4, 3, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 1, 2, 0, 4, 0, 4, 3]

    def build_cc_input(
        system=None,
        position=None,
        seat_position=None,
        fog_light_position=None,
        window_state_target=None,
        window_state_initial=None,
        onoff_state_target=None,
        onoff_state_initial=None,
        head_lights_mode_target=None,
        head_lights_mode_initial=None,
        climate_temperature_value_target=None,
        climate_temperature_value_initial=None,
        seat_heating_level_target=None,
        seat_heating_level_initial=None,

        system2=None,
        position2=None,
        seat_position2=None,
        fog_light_position2=None,
        window_state_target2=None,
        window_state_initial2=None,
        onoff_state_target2=None,
        onoff_state_initial2=None,
        head_lights_mode_target2=None,
        head_lights_mode_initial2=None,
        climate_temperature_value_target2=None,
        climate_temperature_value_initial2=None,
        seat_heating_level_target2=None,
        seat_heating_level_initial2=None,
    ):
        return CCContentInput(
            system=system,
            position=position,
            seat_position=seat_position,
            fog_light_position=fog_light_position,
            window_state_target=window_state_target,
            window_state_initial=window_state_initial,
            onoff_state_target=onoff_state_target,
            onoff_state_initial=onoff_state_initial,
            head_lights_mode_target=head_lights_mode_target,
            head_lights_mode_initial=head_lights_mode_initial,
            climate_temperature_value_target=climate_temperature_value_target,
            climate_temperature_value_initial=climate_temperature_value_initial,
            seat_heating_level_target=seat_heating_level_target,
            seat_heating_level_initial=seat_heating_level_initial,

            system2=system2,
            position2=position2,
            seat_position2=seat_position2,
            fog_light_position2=fog_light_position2,
            window_state_target2=window_state_target2,
            window_state_initial2=window_state_initial2,
            onoff_state_target2=onoff_state_target2,
            onoff_state_initial2=onoff_state_initial2,
            head_lights_mode_target2=head_lights_mode_target2,
            head_lights_mode_initial2=head_lights_mode_initial2,
            climate_temperature_value_target2=climate_temperature_value_target2,
            climate_temperature_value_initial2=climate_temperature_value_initial2,
            seat_heating_level_target2=seat_heating_level_target2,
            seat_heating_level_initial2=seat_heating_level_initial2
        )
    
    test = build_cc_input(
        system="windows",
        #position="driver",
        window_state_target="open",

        system2="climate",
        climate_temperature_value_target2=22
    )

    print(test)
    print("")

    result = ipa.resample_content(
        featurehandler,
        categorical,
        ordinal,
        test)
    print(result)
    print(categorical)
