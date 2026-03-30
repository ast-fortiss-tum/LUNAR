import logging
import json
from json_repair import repair_json
from openai import AzureOpenAI
from dateutil import parser
import os
from user_sim.utils.utilities import check_keys
from dotenv import load_dotenv

import time
import json
import logging

load_dotenv()
# check_keys(["OPENAI_API_KEY"])

client = AzureOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"])

logger = logging.getLogger('Info Logger')

class DataExtraction:

    def __init__(self, conversation, variable_name, dtype, description):

        self.conversation = f"{conversation['interaction']}"
        self.dtype = dtype
        self.variable = variable_name
        self.description = description
        self.prompt = f"""
        You're an assistant that analyzes a conversation between a user and a chatbot.
        Your objective is to test the chatbot's capabilities by extract the information only if the chatbot provides it 
        or verifies it. Output only the requested data, If you couldn't find it, output None.
        "Output format:\n"
        {{ "answer": <value or null> }}
        """

        self.system_message = [
            {"role": "system",
             "content": self.prompt},
            {"role": "user",
             "content": self.conversation}
        ]

    @staticmethod
    def data_process(text, dtype):
        logger.info(f'input text on data process for casting: {text}')

        if text is None or text == 'null':
            return text
        try:
            if dtype == 'int':
                return int(text)
            elif dtype == 'float':
                return float(text)
            elif dtype == 'money':
                return text
            elif dtype == 'str':
                return str(text)
            elif dtype == 'bool':
                return bool(text)
            elif dtype == 'time':
                time = parser.parse(text).time().strftime("%H:%M:%S")
                return time
            elif dtype == 'date':
                date = parser.parse(text).date()
                return date
            else:
                return text

        except ValueError as e:
            logger.warning(f"Error in casting: {e}. Returning 'str({str(text)})'.")
            return str(text)

    def get_data_prompt(self):

        data_type = {'int': 'integer',
                     'float': 'number',
                     'money': 'string',
                     'str': "string",
                     'time': 'string',
                     'bool': 'boolean',
                     'date': 'string'}

        data_format = {'int': '',
                       'float': '',
                       'money': 'Output the data as money with the currency used in the conversation',
                       'str': "Extract and  display concisely only the requested information "
                              "without including additional context",
                       'time': 'Output the data in a "hh:mm:ss" format',
                       'date': 'Output the data in a date format understandable for Python with this structure: dd/mm/yyyy'}

        prompt_type = data_type.get(self.dtype)
        d_format = data_format.get(self.dtype)
        return prompt_type, d_format

    def get_data_extraction(self):

        dtype = self.get_data_prompt()[0]
        dformat = self.get_data_prompt()[1]

        if dtype is None:
            logger.warning(f"Data type {self.dtype} is not supported. Using 'str' by default.")
            dtype = 'string'

        if dformat is None:
            logger.warning(f"Data format for {self.dtype} is not supported. Using default format.")
            dformat = "Extract and  display concisely only the requested information without including additional context"
        
        # response_format = {
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "data_extraction",
        #         "strict": True,
        #         "schema": {
        #             "type": "object",
        #             "required": ["answer"],
        #             "additionalProperties": False,
        #             "properties": {
        #                 "answer": {
        #                     "type": [dtype, 'null'],
        #                     "description": f"{self.description}. {dformat}"
        #                 }
        #             }
        #         }
        #     }
        # }

        max_retries = 3
        retry_delay = 1  # seconds between attempts

        for attempt in range(1, max_retries + 1):
            try:
                # Call the Azure OpenAI API
                response = client.chat.completions.create(
                    model="gpt-5-chat",
                    messages=self.system_message
                )
                results = response.choices[0].message.content
                json_response = repair_json(results)
                payload = json.loads(json_response)
                print("Raw LLM response:", response.choices[0].message.content)
                value = self.data_process(payload["answer"], self.dtype)
                break

            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    logger.error("Max retries reached. Using default value.")
                    value = None
                else:
                    time.sleep(retry_delay)
        return {self.variable: value}
