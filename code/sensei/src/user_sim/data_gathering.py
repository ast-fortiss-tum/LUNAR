import ast
import time
import json
import re
import os
import logging
import pandas as pd

from dotenv import load_dotenv
from openai import AzureOpenAI
from json_repair import repair_json

# ---------------------------------------------------------------------
# Environment and client setup
# ---------------------------------------------------------------------

load_dotenv()

client = AzureOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"]
)

logger = logging.getLogger("Info Logger")

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def extract_dict(in_val):
    reg_ex = r"\{[^{}]*\}"
    coincidence = re.search(reg_ex, in_val, re.DOTALL)
    return coincidence.group(0) if coincidence else None


def to_dict(in_val):
    try:
        dictionary = ast.literal_eval(extract_dict(in_val))
    except (ValueError, SyntaxError) as e:
        logger.error(
            f"Bad dictionary generation: {e}. Setting empty dictionary value."
        )
        dictionary = {}
    return dictionary

# ---------------------------------------------------------------------
# Chatbot assistant
# ---------------------------------------------------------------------

class ChatbotAssistant:
    """
    Determines, for each topic in ask_about, whether it has been answered
    or confirmed by the assistant in a conversation.
    Output structure is enforced via prompting (Azure OpenAI compatible).
    """

    def __init__(self, ask_about):

        # Normalize topic names into JSON keys
        self.ask_about_keys = self.process_ask_about(ask_about)

        keys_str = ", ".join(self.ask_about_keys)

        # System message with explicit structural requirements
        self.system_message = {
            "role": "system",
            "content": (
                "You analyze a conversation and determine whether specific topics "
                "have been answered or confirmed by the assistant, NOT by the user.\n\n"
                "You MUST output a single valid JSON object.\n\n"
                "For EACH of the following topics:\n"
                f"keys: {keys_str}\n\n"
                "The value MUST be an object with EXACTLY these fields:\n"
                "- verification: boolean\n"
                "- data: string or null\n\n"
                
                "Rules:\n"
                "1. Every key MUST be present.\n"
                "2. verification MUST always be present.\n"
                "3. data MUST always be present (use null if not applicable).\n"
                "4. If the topic is not answered or confirmed, set "
                "verification=false and data=null.\n"
                "5. Do NOT omit any key.\n"
                "6. Do NOT add extra fields.\n"
                "7. Output JSON only. No explanations."
                "8. The topic has to be answered by the assistant."


                "Example Output:\n"
                "Topics: ask_for_a_cafe, has_contactless_payment, ask_for_rating\n"
                """
                {{
                    "ask_for_a_poi_being_cafe": {{
                        "verification": true,
                        "data": "cafe"
                    }},
                    "poi_has_contactless_payment": {{
                        "verification": false,
                        "data": null // because the chatbot did not confirm or provide this information
                    }},
                    "ask_for_rating_of_poi": {{
                        "verification": true,
                        "data": "4.5"
                    }}
                }}
                """
            )
        }

        self.messages = [self.system_message]
        self.gathering_register = []

    @staticmethod
    def process_ask_about(ask_about):
        """
        Converts topic strings into JSON-safe keys.
        """
        return [ab.replace(" ", "_") for ab in ask_about]

    def add_message(self, history):
        """
        Adds the conversation history as a single user message and
        immediately triggers dataframe creation.
        """
        text = ""
        for entry in history["interaction"]:
            for speaker, message in entry.items():
                text += f"{speaker}: {message}\n"

        user_message = {
            "role": "user",
            "content": text
        }

        self.messages = [self.system_message, user_message]
        self.gathering_register = self.create_dataframe()
        # assuming df is your dataframe
        os.makedirs("./tmp", exist_ok=True)  # creates folder if missing
        temp_file_path = "./tmp/gathering_register.csv"  # Unix-style temp path
        self.gathering_register.to_csv(temp_file_path, index=False)
        print(f"Dataframe written to {temp_file_path}")
        
    def get_json(self):
        """
        Calls Azure OpenAI and returns the parsed JSON object.
        """
        max_retries = 3
        failed = False  

        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model="gpt-5-chat",  # Azure deployment name
                    messages=self.messages,
                    response_format={"type": "json_object"}
                )
                break
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    failed = True
                time.sleep(1)

        if failed:
            # Fallback behavior: return a valid object matching the expected schema
            logger.error("Failed to get valid response from LLM after multiple attempts; returning fallback.")
            return {
                key: {"verification": False, "data": None}
                for key in self.ask_about_keys
            }

        # At this point, response exists
        try:
            # If response is a string, use it directly
            if isinstance(response, str):
                result = response
            else:
                result = response.choices[0].message.content

            result = repair_json(result)  # Ensure valid JSON string
            return json.loads(result)

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return fallback if parsing fails
            return {
                key: {"verification": False, "data": None}
                for key in self.ask_about_keys
            }

    def create_dataframe(self):
        """
        Normalizes the model output into a pandas DataFrame.
        """
        data_dict = self.get_json()
        # print("Data dict received from LLM:", data_dict)  # Debugging output
        df_long = pd.DataFrame.from_dict(data_dict, orient="index").reset_index()
        df_long = df_long.rename(columns={"index": "topic"})
        return df_long
