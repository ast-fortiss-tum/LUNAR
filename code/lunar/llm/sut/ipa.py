import json
import traceback

import numpy as np

from examples.navi.models import NaviContentInput
from examples.navi.navi_utterance_generator import NaviUtteranceGenerator
from llm.features.feature_handler import FeatureHandler
from llm.llms import LLMType
from typing import List, Optional
from llm.sut.ipa_base import IPABase
from llm.model.qa_simout import QASimulationOutput
from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.model.models import Utterance, Conversation, Turn
from llm.model.conversation_intents import UserIntent
from llm.llms import pass_llm
from llm.config import LLM_IPA
import logging as log
from llm.prompts import SYSTEM_PROMPT, SYSTEM_ANSWER_PROMPT


class IPA(IPABase):
    memory: List = []
    ipa_name = "generic"
    global_user_counter = 0

    @staticmethod
    def send_utterance_request(
        request,
        specific_user_id=None,
        max_retries: int = 3,
        system_message: str = SYSTEM_PROMPT,
        temperature: float = 0.7,
        context: object = None,
    ) -> dict:
        for attempt in range(max_retries):
            try:
                if specific_user_id:
                    user_id_val = specific_user_id
                else:
                    user_id_val = f"user_{IPA.global_user_counter}"
                    IPA.global_user_counter += 1

                print("user_id for request:", user_id_val)

                response = pass_llm(
                    msg=request,
                    llm_type=LLMType(LLM_IPA),
                    temperature=temperature,
                    system_message=system_message,
                    context=context,
                )
                return {
                    "data": {
                        "result": response,
                        "los": [],
                    }
                }
            except Exception:
                print(f"Attempt {attempt + 1} failed with error:")
                traceback.print_exc()

        return {"data": {"result": "Request after max retries failed.", "los": []}}

    @staticmethod
    def simulate(
        list_individuals: List[List[Utterance]],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        max_retries: int = 5,
        llm_type=LLMType(LLM_IPA),
        system_message: str = SYSTEM_PROMPT,
    ) -> List[QASimulationOutput]:

        results = []
        log.info(f"[IPA] list_individuals: {list_individuals}")

        for utterance in list_individuals:
            utterance = utterance[0]

            def check_utterance_in_mem(utt):
                for u in IPA.memory:
                    if u.question == utt.question:
                        return True, u
                return False, utt

            in_memory, memory_utterance = check_utterance_in_mem(utterance)

            if not in_memory:
                for attempt in range(max_retries):
                    try:
                        response = pass_llm(
                            msg=utterance.question,
                            llm_type=llm_type,
                            temperature=temperature,
                            context=context,
                            system_message=system_message,
                        )
                        if response is not None:
                            utterance.answer = response
                            utterance.raw_output = {"system_output": response}
                            break
                    except Exception as e:
                        traceback.print_exc()
                        print(
                            f"[IPA] Attempt {attempt + 1} failed with error: {e} "
                            f"using llm: {LLMType(LLM_IPA)}"
                        )
                IPA.memory.append(utterance)
            else:
                print("Utterance already in memory")
                utterance.answer = memory_utterance.answer

            result = QASimulationOutput(
                utterance=utterance,
                model=LLMType(LLM_IPA),
                ipa=IPA.ipa_name,
            )
            results.append(result)

        return results

    @staticmethod
    def simulate_turn(
        user_text: str,
        user_intent: str,
        user_id: str,
        current_content_input: Optional[NaviContentInput],
        history: List[str],
        max_retries: int = 3,
        **kwargs,
    ) -> Turn:
        """
        Process a single turn via generic LLM call.
        """
        context = kwargs.get("context")
        conversation: Optional[Conversation] = kwargs.get("conversation")

        effective_system_message = SYSTEM_ANSWER_PROMPT.format(
            context=context,
            history=conversation.get_dialogue_history_str() if conversation else "",
        )

        response_obj = IPA.send_utterance_request(
            user_text,
            specific_user_id=user_id,
            max_retries=max_retries,
            system_message=effective_system_message,
            context=context,
        )

        sys_answer_text = response_obj.get("data", {}).get("result", "System Error")

        turn = Turn(
            question=user_text,
            answer=sys_answer_text,
            question_intent=user_intent,
            content_input=current_content_input.model_copy() if current_content_input else None,
            content_output_list=[],
            poi_exists=False,
            raw_output=response_obj,
        )

        history.append(f"User: {user_text}")
        history.append(f"System: {sys_answer_text}")

        return turn

    @staticmethod
    def simulate_conversation(
        list_individuals: List[Conversation],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float,
        time_step: float = 10,
        do_visualize: bool = False,
        temperature: float = 0,
        context: object = None,
        config_path: str = "configs/features_simple_judge_industry.json",
        max_retries: int = 3,
        min_turns: int = 2,
        max_turns: int = 5,
    ) -> List[MultiTurnSimulationOutput]:

        feature_handler = FeatureHandler.from_json(config_path)
        utterance_gen = NaviUtteranceGenerator(feature_handler=feature_handler)
        llm_type = LLMType(LLM_IPA)

        results = []

        for conversation_wrapper in list_individuals:
            conversation = conversation_wrapper[0]

            user_id = f"user_{IPA.global_user_counter}"
            IPA.global_user_counter += 1
            conversation.assigned_user_id = user_id

            conversation = IPA.run_conversation_loop(
                conversation=conversation,
                feature_handler=feature_handler,
                utterance_gen=utterance_gen,
                llm_type=llm_type,
                context=context,
                max_retries=max_retries,
                min_turns=min_turns,
                max_turns=max_turns,
            )

            result = MultiTurnSimulationOutput(
                conversation=conversation,
                model=llm_type,
                ipa=IPA.ipa_name,
            )
            results.append(result)

        return results


# class CustomEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Utterance):
#             return obj.to_dict()
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super().default(obj)
