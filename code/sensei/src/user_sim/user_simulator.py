import logging
from typing import List
import sys
from pathlib import Path
import site

for name in ["opensbt-llm", "OpenSBT-LLM"]:
    base = Path("..") / name
    if base.exists():
        sys.path.insert(0, str(base))
        site = base / "venv/lib/python3.11/site-packages"
        if site.exists():
            sys.path.insert(0, str(site))
        break

sys.path.insert(0,"/home/q680122/.pyenv/versions/3.11.8/envs/myvenv/lib/python3.11/site-packages/")

from user_sim.venue_match_extraction import VenueMatchExtraction

from .data_extraction import DataExtraction
from .utils.config import errors
from .utils.utilities import *
from .data_gathering import *
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
from llm.llms import pass_llm, LLMType
import time
import traceback
import random

parser = StrOutputParser()
logger = logging.getLogger("Info Logger")


class LLMCallHandler:
    """
    OpenSBT-style LLM calling:
      - builds the same prompt as PromptTemplate would
      - calls pass_llm(prompt)
      - parses with StrOutputParser
    """
    def __init__(self, user_profile, parser_obj):
        self.user_profile = user_profile
        self.parser = parser_obj

        self.user_role_prompt = PromptTemplate(
            input_variables=["reminder", "history"],
            template=self.set_role_template(),
        )

    def set_role_template(self):
        reminder = "{reminder}"
        history = "History of the conversation so far: {history}"
        return self.user_profile.role + reminder + history

    def invoke(self, reminder: str, history: str, llm_type: str) -> str:
        print("using model:", llm_type)
        prompt_str = self.user_role_prompt.format(reminder=reminder, history=history)

        last_exc: Exception | None = None

        for attempt in range(1, 6):  # max repeat 5 times
            try:
                response_text = pass_llm(prompt_str, llm_type=llm_type)

                # retry on empty model output
                if response_text is None or not str(response_text).strip():
                    continue

                parsed = self.parser.parse(response_text)

                # retry on empty parsed output
                if parsed is None or not str(parsed).strip():
                    continue

                return parsed

            except Exception as e:
                last_exc = e
                print(f"Failure occurred calling LLM (attempt {attempt}/5). Stack trace:")
                traceback.print_exc()  # prints the current exception's full stack trace to stderr
                # retry on failure
                continue
            finally:
                # optional tiny delay to avoid hammering the backend
                if attempt < 5:
                    time.sleep(0.1)

        # if all retries failed: return empty output (as you requested)
        # (If you prefer raising instead, tell me.)
        return ""
class UserGeneration:
    def __init__(self, user_profile, chatbot, user_id=1):
        self.user_profile = user_profile
        self.chatbot = chatbot

        self.conversation_history = {"interaction": []}
        self.ask_about = user_profile.ask_about.prompt()
        self.data_gathering = ChatbotAssistant(user_profile.ask_about.phrases)

        self.goal_style = user_profile.goal_style
        self.test_name = user_profile.test_name

        self.repeat_count = 0
        self.loop_count = 0
        self.interaction_count = 0

        self.my_context = self.InitialContext()
        self.output_slots = self.__build_slot_dict()
        self.error_report = []

        # OpenSBT new llm call path (no intents)
        self.wrapper_pass_llm = LLMCallHandler(user_profile, parser)

        # --- keep your init vars (optional, no intent logic) ---
        self.user_id = user_id
        self.phrases_per_turn = []
        self.variables_per_turn = []
        self.retrieved_objs_per_turn = []
        # ------------------------------------------------------

    def __build_slot_dict(self):
        slot_dict = {}
        for output in self.user_profile.output:
            var_name = list(output.keys())[0]
            slot_dict[var_name] = None
        return slot_dict

    class InitialContext:
        def __init__(self):
            self.original_context = []
            self.context_list = []

        def initiate_context(self, context):
            default_context = [
                "never recreate a whole conversation, just act as you're a user or client",
                "never indicate that you are the user, like 'user: bla bla'",
                "Sometimes, interact with what the assistant just said.",
                "Never act as the assistant, always behave as a user.",
                "Don't end the conversation until you've asked everything you need.",
                "you're testing a chatbot, so there are random values or irrational things in your requests",
            ]
            if isinstance(context, list):
                self.original_context = context.copy() + default_context.copy()
                self.context_list = context.copy() + default_context.copy()
            else:
                self.original_context = [context] + default_context
                self.context_list = [context] + default_context

        def add_context(self, new_context):
            if isinstance(new_context, list):
                self.context_list.extend(new_context)
            else:
                self.context_list.append(new_context)

        def get_context(self):
            return ". ".join(self.context_list)

        def reset_context(self):
            self.context_list = self.original_context.copy()

    def repetition_track(self, response, reps=3):
        self.my_context.reset_context()
        logger.info(f"Context list: {self.my_context.context_list}")

        if nlp_processor(response, self.chatbot.fallback, 0.6):
            self.repeat_count += 1
            self.loop_count += 1
            logger.info(f"is fallback. Repeat_count: {self.repeat_count}. Loop count: {self.loop_count}")

            if self.repeat_count >= reps:
                self.repeat_count = 0
                change_topic = """
                               Since the assistant is not understanding what you're saying, change the 
                               topic to other things to ask about without starting a new conversation
                               """
                self.my_context.add_context(change_topic)
            else:
                ask_repetition = """
                                If the assistant asks you to repeat the question, repeat the last question the user 
                                said but rephrase it.
                                """
                self.my_context.add_context(ask_repetition)
        else:
            self.repeat_count = 0
            self.loop_count = 0

    @staticmethod
    def conversation_ending(response):
        return nlp_processor(response, "src/testing/user_sim/end_conversation_patterns.yml", 0.5)

    def get_history(self):
        lines = []
        for inp in self.conversation_history["interaction"]:
            for k, v in inp.items():
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def update_history(self, role, message):
        self.conversation_history["interaction"].append({role: message})
   
    def match_provided(self):
        verifier = VenueMatchExtraction(self.conversation_history)
        verdict = verifier.detect()
        print("Venue match verdict:", verdict)
        # print(verdict["match"], verdict["venue_name"], verdict["venue_type"])
        return verdict["match"]

    def end_conversation(self, input_msg):
        if self.goal_style[0] in ("steps", "random steps"):
            return self.interaction_count >= self.goal_style[1]

        if self.conversation_ending(input_msg) or self.loop_count >= 9:
            errors.append({1000: "Exceeded loop Limit"})
            logger.warning("Loop count surpassed 9 interactions. Ending conversation.")
            return True

        if "all_answered" in self.goal_style[0] or "default" in self.goal_style[0]:
            print("self.goal_style:", self.goal_style)
            # rand = random.random()
            if (
                (self.data_gathering.gathering_register["verification"].all() and \
                self.all_data_collected())
                or self.goal_style[2] <= self.interaction_count
            ):
            # if (
            #     (self.match_provided() and rand < 0.5) \
            #         #self.data_gathering.gathering_register["verification"].all() and 
            #     # self.all_data_collected())
            #     or self.goal_style[2] <= self.interaction_count
            # ):
                logger.info(f"limit amount of interactions achieved: {self.goal_style[2]}. Ending conversation.")
                return True
            return False

        return False

    def all_data_collected(self):
        for output in self.user_profile.output:
            var_name = list(output.keys())[0]
            var_dict = output.get(var_name)

            if var_name in self.output_slots and self.output_slots[var_name] is not None:
                continue

            my_data_extract = DataExtraction(
                self.conversation_history,
                var_name,
                var_dict["type"],
                var_dict["description"],
            )
            value = my_data_extract.get_data_extraction()
            # value = "mocked"
            if value[var_name] is None:
                return False

            self.output_slots[var_name] = value[var_name]

        return True

    def get_response(self, input_msg, llm_type="gpt-4o-mini"):
        self.update_history("Assistant", input_msg)
        self.data_gathering.add_message(self.conversation_history)

        if self.end_conversation(input_msg):
            return "exit"

        self.repetition_track(input_msg)
        self.my_context.add_context(self.user_profile.get_language())

        history = self.get_history()
        user_response = self.wrapper_pass_llm.invoke(
            reminder=self.my_context.get_context(),
            history=history,
            llm_type=LLMType.DEEPSEEK_V3_0324
        )

        self.update_history("User", user_response)
        self.interaction_count += 1

        # optional bookkeeping, no intent logic
        self.variables_per_turn.append({})
        self.phrases_per_turn.append(None)

        return user_response

    def get_interaction_styles_prompt(self):
        interaction_style_prompt = []
        for instance in self.user_profile.interaction_styles:
            if not instance.change_language_flag:
                interaction_style_prompt.append(instance.get_prompt())
        return "".join(interaction_style_prompt)

    def open_conversation(self, input_msg=None):
        interaction_style_prompt = self.get_interaction_styles_prompt()
        self.my_context.initiate_context([self.user_profile.context, interaction_style_prompt, self.ask_about])
        self.my_context.add_context(self.user_profile.get_language())

        history = self.get_history()

        if input_msg:
            self.update_history("Assistant", input_msg)
            self.data_gathering.add_message(self.conversation_history)
            if self.end_conversation(input_msg):
                return "exit"
            self.repetition_track(input_msg)

        user_response = self.wrapper_pass_llm.invoke(
            reminder=self.my_context.get_context(),
            history=history,
            llm_type=LLMType.DEEPSEEK_V3_0324
        )

        self.update_history("User", user_response)
        self.data_gathering.add_message(self.conversation_history)
        self.interaction_count += 1

        # optional bookkeeping, no intent logic
        self.variables_per_turn.append({})
        self.phrases_per_turn.append(self.ask_about)

        return user_response