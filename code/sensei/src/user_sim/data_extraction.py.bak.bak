import os
import json
import logging
from dateutil import parser
from dotenv import load_dotenv
from openai import AzureOpenAI

# ---------------------------------------------------------------------
# Environment & client setup
# ---------------------------------------------------------------------

load_dotenv()

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "AZURE_ENDPOINT",
    "OPENAI_API_VERSION",
]

for key in REQUIRED_ENV_VARS:
    if key not in os.environ:
        raise EnvironmentError(f"Missing environment variable: {key}")

client = AzureOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataExtraction")

# ---------------------------------------------------------------------
# DataExtraction class
# ---------------------------------------------------------------------

class DataExtraction:
    """
    Extracts a single variable from a conversation using Azure OpenAI
    chat completions with JSON-only output and post-hoc validation.
    """

    def __init__(self, conversation, variable_name, dtype, description):
        """
        Parameters
        ----------
        conversation : dict
            Must contain key 'interaction' with conversation text
        variable_name : str
            Name of the extracted variable
        dtype : str
            One of: int, float, money, str, bool, time, date
        description : str
            Description of the variable to extract
        """
        self.conversation = conversation["interaction"]
        self.variable = variable_name
        self.dtype = dtype
        self.description = description

        self.base_system_prompt = (
            "You are an assistant that extracts structured information "
            "from a conversation between a user and a chatbot.\n"
            "Return ONLY a valid JSON object with exactly one key named 'answer'.\n"
            "If the information is not explicitly stated or confirmed, "
            "set 'answer' to null.\n"
            "Do not include explanations, comments, or extra keys."
        )

    # -----------------------------------------------------------------
    # Type casting
    # -----------------------------------------------------------------

    @staticmethod
    def data_process(value, dtype):
        logger.info(f"Raw extracted value: {value}")

        if value is None:
            return None

        try:
            if dtype == "int":
                return int(value)
            elif dtype == "float":
                return float(value)
            elif dtype == "bool":
                return bool(value)
            elif dtype == "time":
                return parser.parse(value).time().strftime("%H:%M:%S")
            elif dtype == "date":
                return parser.parse(value).date()
            elif dtype in {"money", "str"}:
                return str(value)
            else:
                logger.warning(f"Unsupported dtype '{dtype}', returning raw value.")
                return value
        except Exception as e:
            logger.warning(f"Type casting failed: {e}")
            return None

    # -----------------------------------------------------------------
    # Prompt helpers
    # -----------------------------------------------------------------

    def _format_instruction(self):
        format_map = {
            "int": "Return an integer.",
            "float": "Return a numeric value.",
            "money": "Return the monetary amount, including currency if mentioned.",
            "str": "Return a concise string.",
            "bool": "Return true or false.",
            "time": "Return time in HH:MM:SS format.",
            "date": "Return date in a Python-readable date format.",
        }
        return format_map.get(self.dtype, "Return the value as a string.")

    # -----------------------------------------------------------------
    # Main extraction call
    # -----------------------------------------------------------------

    def get_data_extraction(self):
        extraction_prompt = (
            f"{self.description}\n"
            f"{self._format_instruction()}\n\n"
            "Output format:\n"
            '{ "answer": <value or null> }'
        )
        conversation_text = "\n".join(
                f"User: {turn['User']}" if 'User' in turn else f"Assistant: {turn['Assistant']}"
                for turn in self.conversation
        )
        messages = [
            {"role": "system", "content": self.base_system_prompt},
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": conversation_text},
        ]

        # print("Conversation:\n", self.conversation)  # Debugging

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Azure deployment name
            messages=messages,
            response_format={"type": "json_object"},
        )

        logger.info("LLM response received.")

        try:
            payload = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON returned: {e}")
            return {self.variable: None}

        if not isinstance(payload, dict):
            logger.warning("Response is not a JSON object.")
            return {self.variable: None}

        if "answer" not in payload:
            logger.warning("Missing 'answer' key in response.")
            return {self.variable: None}

        value = self.data_process(payload["answer"], self.dtype)
        return {self.variable: value}


# ---------------------------------------------------------------------
# Example usage (can be removed in production)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    conversation_example = {
        "interaction": (
            "The pizza will arrive in 30 minutes and will cost 12.50 euros."
        )
    }

    extractor = DataExtraction(
        conversation=conversation_example,
        variable_name="delivery_time",
        dtype="time",
        description="Extract the delivery time stated by the chatbot."
    )

    result = extractor.get_data_extraction()
    print(result)
