import logging as log
import random
import time
from enum import Enum
from typing import Dict

from llm.config import DEBUG, MAX_TOKENS


class LLMType(Enum):
    MOCK = "mock"
    GPT_3O_MINI = "gpt-3o-mini"
    GPT_35_TURBO = "gpt-35-turbo"
    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_CHAT = "gpt-5-chat"
    LLAMA3_2 = "llama3.2"
    DOLPHIN_MISTRAL = "dolphin-mistral"
    DEEPSEEK_V2 = "deepseek-v2"
    GPT_OSS = "gpt-oss:20b"
    QWEN = "qwen"
    HF = "hf"
    MISTRAL="mistral"
    GEMMA="gemma"
    MISTRAL_7B_INSTRUCT_V02_GPTQ = "Mistral-7B-Instruct-v0.2-GPTQ"
    DEEPSEEK_R1_QCBAR = "DeepSeek-R1-qcbar"
    DEEPSEEK_V3_0324 = "DeepSeek-V3-0324"
    DEEPSEEK_V3_0324_2 = "DeepSeek-V3-0324-2"
    DOLPHIN_21_UNCENSORED = "mainzone/dolphin-2.1-mistral-7b-uncensored"
    DOLPHIN3 = "dolphin3"
    GEMINI_3_FLASH = "gemini-3-flash-preview"
    GEMINI_3_PRO = "gemini-3-pro-preview"
    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    CLAUDE_35_SONNET = "claude-35-sonnet"
    CLAUDE_37_SONNET = "claude-37-sonnet"
    CLAUDE_4_SONNET = "claude-4-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"


ALL_MODELS = [llm.value for llm in LLMType]
GPT5_MODELS = {LLMType.GPT_5, LLMType.GPT_5_MINI,  LLMType.GPT_5_NANO, LLMType.GPT_5_CHAT}
LOCAL_MODELS = {LLMType.QWEN, LLMType.GEMMA, LLMType.MISTRAL, LLMType.LLAMA3_2, LLMType.DOLPHIN_MISTRAL, LLMType.DEEPSEEK_V2, LLMType.DOLPHIN3}
OPENAI_MODELS = GPT5_MODELS | {LLMType.GPT_35_TURBO, LLMType.GPT_4, LLMType.GPT_4O, LLMType.GPT_4_1, LLMType.GPT_4O_MINI, LLMType.GPT_3O_MINI}
DEEPSEEK_MODELS = {LLMType.DEEPSEEK_V3_0324_2, LLMType.DEEPSEEK_V3_0324, LLMType.DEEPSEEK_R1_QCBAR}
GEMINI_MODELS = {LLMType.GEMINI_3_FLASH, LLMType.GEMINI_3_PRO, LLMType.GEMINI_25_PRO, LLMType.GEMINI_25_FLASH}

GEMINI_THINKING_MODELS = {LLMType.GEMINI_25_PRO, LLMType.GEMINI_25_FLASH, LLMType.GEMINI_3_PRO, LLMType.GEMINI_3_FLASH}
ANTHROPIC_MODELS = {LLMType.CLAUDE_35_SONNET, LLMType.CLAUDE_37_SONNET, LLMType.CLAUDE_4_SONNET, LLMType.CLAUDE_3_HAIKU}


class ModelStatistics:
    # initialize statistics for each model, including a thinking_tokens counter
    statistics = {
        model: {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
            "time": 0,
            "calls": 0,
            "costs": {"input": 0.0, "output": 0.0, "thinking": 0.0, "total": 0.0},
        }
        for model in LLMType
    }

    LLM_COST_RATES = {
        # Costs per 1k tokens; default is 0 for unlisted models.
        # "thinking" rate applies to thoughts_token_count on gemini thinking models;
        # google bills these at the same rate as output tokens.

        LLMType.GPT_4: {"input": 0.01, "output": 0.03, "thinking": 0.0},
        LLMType.GPT_4O: {"input": 0.0025, "output": 0.010, "thinking": 0.0},
        LLMType.GPT_4O_MINI: {"input": 0.00015, "output": 0.0006, "thinking": 0.0},
        LLMType.GPT_35_TURBO: {"input": 0.0005, "output": 0.0015, "thinking": 0.0},
        LLMType.GPT_5: {"input": 0.00125, "output": 0.010, "thinking": 0.0},
        LLMType.GPT_5_CHAT: {"input": 0.00125, "output": 0.010, "thinking": 0.0},
        LLMType.GPT_5_MINI: {"input": 0.00025, "output": 0.002, "thinking": 0.0},
        LLMType.GPT_5_NANO: {"input": 0.00005, "output": 0.00040, "thinking": 0.0},

        LLMType.DEEPSEEK_R1_QCBAR: {"input": 0.001485, "output": 0.00594, "thinking": 0.0},
        LLMType.DEEPSEEK_V3_0324: {"input": 0.00114, "output": 0.00456, "thinking": 0.0},

        # gemini 2.5 / 3 thinking models: thinking tokens billed at output rate
        LLMType.GEMINI_3_FLASH: {"input": 0.0005, "output": 0.003, "thinking": 0.003},
        LLMType.GEMINI_3_PRO: {"input": 0.002, "output": 0.012, "thinking": 0.012},
        LLMType.GEMINI_25_PRO: {"input": 0.00125, "output": 0.010, "thinking": 0.010},
        LLMType.GEMINI_25_FLASH: {"input": 0.0003, "output": 0.0025, "thinking": 0.0025},

        LLMType.CLAUDE_35_SONNET: {"input": 0.003, "output": 0.015, "thinking": 0.0},
        LLMType.CLAUDE_37_SONNET: {"input": 0.003, "output": 0.015, "thinking": 0.0},
        LLMType.CLAUDE_4_SONNET: {"input": 0.003, "output": 0.015, "thinking": 0.0},
        LLMType.CLAUDE_3_HAIKU: {"input": 0.00025, "output": 0.00125, "thinking": 0.0},
    }

    @classmethod
    def record_usage(cls, model_type, input_tokens, output_tokens, time_taken, thinking_tokens=0):
        model_stats = cls.statistics[model_type]
        model_stats["input_tokens"] += input_tokens
        model_stats["output_tokens"] += output_tokens
        model_stats["thinking_tokens"] += thinking_tokens
        # total_tokens tracks all billable tokens: input + visible output + thinking
        model_stats["total_tokens"] += input_tokens + output_tokens + thinking_tokens
        model_stats["time"] += time_taken
        model_stats["calls"] += 1

        rates = cls.LLM_COST_RATES.get(model_type, {"input": 0, "output": 0, "thinking": 0})

        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        thinking_cost = (thinking_tokens / 1000) * rates.get("thinking", 0)
        total_cost = input_cost + output_cost + thinking_cost

        model_stats["costs"]["input"] += input_cost
        model_stats["costs"]["output"] += output_cost
        model_stats["costs"]["thinking"] += thinking_cost
        model_stats["costs"]["total"] += total_cost

    @classmethod
    def complete_statistics(cls) -> Dict:
        result = {}
        for llm_type, base_stats in cls.statistics.items():
            calls = base_stats["calls"]
            result[llm_type.value] = {
                **base_stats,
                "average call time": (base_stats["time"] / calls if calls > 0 else 0),
            }
        return result
    
    @classmethod 
    def get_statistics(cls, model_type): 
        return cls.statistics.get(model_type, 
            {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
            "time": 0,
            "calls": 0,
            "costs": {"input": 0.0, "output": 0.0, "thinking": 0.0, "total": 0.0},
            }
        )

    @classmethod
    def total_values(cls) -> Dict:
        stats = cls.complete_statistics().values()
        return {
            "input_tokens": sum(s["input_tokens"] for s in stats),
            "output_tokens": sum(s["output_tokens"] for s in stats),
            "thinking_tokens": sum(s["thinking_tokens"] for s in stats),
            "total_tokens": sum(s["total_tokens"] for s in stats),
            "time": sum(s["time"] for s in stats),
            "calls": sum(s["calls"] for s in stats),
            "costs": {
                "input": sum(s["costs"]["input"] for s in stats),
                "output": sum(s["costs"]["output"] for s in stats),
                "thinking": sum(s["costs"]["thinking"] for s in stats),
                "total": sum(s["costs"]["total"] for s in stats),
            },
            "average call time": (
                sum(s["time"] for s in stats) / sum(s["calls"] for s in stats)
                if sum(s["calls"] for s in stats) > 0
                else 0
            ),
        }

def pass_llm(
    msg,
    max_tokens=MAX_TOKENS,
    temperature=0,
    llm_type=LLMType.GPT_4O,
    context=None,
    system_message="You are an intelligent system",
):
    prompt = msg
    start_time = time.time()

    if temperature is None:
        temperature = random.random()

    # thinking_tokens is only populated by call_gemini for thinking-capable models - default it to 0
    thinking_tokens = 0

    try:

        if llm_type == LLMType.MOCK:
            response_text, input_tokens, output_tokens, elapsed_time = call_mock(
            prompt, "", max_tokens, temperature, system_message
        )
        elif llm_type == LLMType.HF:
            from llm.call_hf import call_hf_llm
    
            response_text, input_tokens, output_tokens, elapsed_time = call_hf_llm(
                prompt, max_tokens, temperature, system_message, context
            )
        elif llm_type in LOCAL_MODELS:
            from llm.call_ollama import call_ollama
    
            response_text, input_tokens, output_tokens, elapsed_time = call_ollama(
                prompt, max_tokens, temperature, llm_type.value, system_message, context
            )
        elif llm_type in OPENAI_MODELS:
            from llm.llm_openai import call_openai
    
            response_text, input_tokens, output_tokens, elapsed_time = call_openai(
                prompt,
                max_tokens,
                temperature,
                system_message,
                context,
                model=llm_type.value,
            )
        elif llm_type in DEEPSEEK_MODELS:
            from llm.call_deepseek import call_deepseek
    
            response_text, input_tokens, output_tokens, elapsed_time = call_deepseek(
                llm_type.value, prompt, max_tokens, temperature, system_message, context
            )
        elif llm_type in GEMINI_MODELS:
            from llm.call_gemini import call_gemini

            # Call_gemini returns a 6-tuple: text, input, output, time, thinking, cache - unpack thinking_tokens here; cached_tokens are informational only for now.
            response_text, input_tokens, output_tokens, elapsed_time, thinking_tokens, _cached_tokens = call_gemini(
                llm_type.value, prompt, max_tokens, temperature, system_message, context
            )
        elif llm_type in ANTHROPIC_MODELS:
            from llm.call_anthropic import call_anthropic

            response_text, input_tokens, output_tokens, elapsed_time = call_anthropic(
                llm_type.value, prompt, max_tokens, temperature, system_message, context
            )
        else:
            raise ValueError(
                f"LLM {llm_type} is not supported. List of supported LLMs: "
                + ", ".join([model.name for model in LLMType])
            )
    except ValueError as e:
        raise e
    except Exception as e:
        print("Error in pass_llm\n", e)
        response_text, input_tokens, output_tokens, elapsed_time = "", 0, 0, 0
    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Record usage statistics, including thinking tokens for Gemini thinking models
    ModelStatistics.record_usage(llm_type, input_tokens, output_tokens, elapsed_time, thinking_tokens)

    # Clean up the response text
    if response_text is None:
        response_text = ""
    response_text = response_text.replace('"', '')

    if DEBUG:
        log.info(f"QUESTION: {prompt}")
        log.info(f"ANSWER: {response_text}")

    log.info(
        f"[Overview] LLM {llm_type} calls: {ModelStatistics.get_statistics(llm_type)['calls']}"
    )
    log.info(
        f"[Overview] LLM {llm_type} token usage: {ModelStatistics.get_statistics(llm_type)['total_tokens']}"
    )

    return response_text


def call_mock(prompt, role, max_tokens, temperature, system_message):
    output = f"I am just a mock, a random number is {random.randint(239, 239239)}"
    return output, len(prompt), len(output), 0


if __name__ == "__main__":
    # Define your input
    message = "What is the capital of France?"

    response = pass_llm(
        msg=message,
        llm_type=LLMType.CLAUDE_4_SONNET
    )

    # Print results
    print("Response:", response)
