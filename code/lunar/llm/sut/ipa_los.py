import json
import traceback
from typing import List, Optional

from llm.llms import LLMType, pass_llm
from llm.sut.ipa_base import IPABase
from llm.model.qa_simout import QASimulationOutput
from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.model.models import Utterance, Conversation, Turn
from llm.config import LLM_IPA
import logging as log
from llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_CONTENT_INPUT, SYSTEM_PROMPT_CONTENT_INPUT_HISTORY
from json_repair import repair_json
from llm.features import FeatureHandler
from examples.navi.navi_utterance_generator import NaviUtteranceGenerator
from examples.navi.models import NaviContentInput, NaviContentOutput


class IPA_LOS(IPABase):
    memory: List = []
    ipa_name = "generic_with_los"
    global_user_counter = 0

    @staticmethod
    def send_utterance_request_los(
        request,
        temperature=0.3,
        context=None,
        system_message=SYSTEM_PROMPT_CONTENT_INPUT,
        max_retries: int = 3,
    ):
        attempt = 0
        while attempt < max_retries:
            try:
                response = pass_llm(
                    msg=request,
                    llm_type=LLMType(LLM_IPA),
                    temperature=temperature,
                    context=context,
                    system_message=system_message,
                )

                if response is not None:
                    response = repair_json(response)
                    print("IPA LOS Response:", response)
                    response_parsed = json.loads(response)
                    print("response parsed:", response_parsed)
                    answer = response_parsed["system_response"]

                    assert answer is not None, "System response is None"

                    if len(response_parsed["los"]) > 0:
                        _ = [
                            NaviContentOutput.model_validate(entry)
                            for entry in response_parsed["los"]
                        ]
                    return {
                        "data": {
                            "result": response_parsed["system_response"],
                            "los": response_parsed["los"],
                        }
                    }
            except Exception as e:
                traceback.print_exc()
                print(f"[IPA_LOS] Attempt {attempt + 1} failed with error: {e}")
            attempt += 1

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
        max_retries: int = 3,
        system_message: str = SYSTEM_PROMPT,
    ) -> List[QASimulationOutput]:

        results = []
        log.info(f"[IPA] list_individuals: {list_individuals}")

        for utterance in list_individuals:
            utterance = utterance[0]

            def check_utterance_in_mem(utt):
                for u in IPA_LOS.memory:
                    if u.question == utt.question:
                        return True, u
                return False, utt

            in_memory, memory_utterance = check_utterance_in_mem(utterance)

            if not in_memory:
                IPA_LOS.memory.append(utterance)
            else:
                print("Utterance already in memory")
                utterance.answer = memory_utterance.answer

            print(f"[IPA LOS] {utterance}")
            result = QASimulationOutput(
                utterance=utterance,
                model=LLMType(LLM_IPA),
                ipa=IPA_LOS.ipa_name,
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
        Process a single turn via the LOS-generating LLM.
        """
        context = kwargs.get("context")
        conversation: Optional[Conversation] = kwargs.get("conversation")

        effective_system_message = SYSTEM_PROMPT_CONTENT_INPUT_HISTORY.format(
            context=context,
            history=conversation.get_dialogue_history_str() if conversation else "",
        )

        response_obj = IPA_LOS.send_utterance_request_los(
            user_text,
            system_message=effective_system_message,
            max_retries=max_retries,
        )

        sys_answer_text = response_obj.get("data", {}).get("result", "System Error")

        print("sys_answer_text:", sys_answer_text)

        out_list_raw = response_obj.get("data", {}).get("los", [])
        print("out_list_raw:", out_list_raw)

        if len(out_list_raw) > 0:
            content_output_list = [
                NaviContentOutput.model_validate(entry) for entry in out_list_raw
            ]
        else:
            content_output_list = []

        turn = Turn(
            question=user_text,
            answer=sys_answer_text,
            question_intent=user_intent,
            content_input=current_content_input.model_copy() if current_content_input else None,
            content_output_list=content_output_list,
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
        max_repeats: int = 2,
    ) -> List[MultiTurnSimulationOutput]:

        feature_handler = FeatureHandler.from_json(config_path)
        utterance_gen = NaviUtteranceGenerator(feature_handler=feature_handler)
        llm_type = LLMType(LLM_IPA)

        results = []

        for conversation_wrapper in list_individuals:
            conversation = conversation_wrapper[0]

            user_id = f"user_{IPA_LOS.global_user_counter}"
            IPA_LOS.global_user_counter += 1
            conversation.assigned_user_id = user_id

            conversation = IPA_LOS.run_conversation_loop(
                conversation=conversation,
                feature_handler=feature_handler,
                utterance_gen=utterance_gen,
                llm_type=llm_type,
                context=context,
                max_retries=max_retries,
                min_turns=min_turns,
                max_turns=max_turns,
                max_repeats=max_repeats,
            )

            result = MultiTurnSimulationOutput(
                conversation=conversation,
                model=llm_type,
                ipa=IPA_LOS.ipa_name,
            )
            results.append(result)

        return results