import numpy as np
from judge_eval.prompts import PROMT_EVALUATE_REQUEST_RESPONSE_HELP
from llm.config import DEBUG, LLM_VALIDATOR
import random
from llm.prompts import VALIDATION_PROMPT, VALIDATION_PROMPT_RAW_OUTPUT
from llm.llms import LLMType, pass_llm
import json
from json_repair import repair_json

def llm_validator(
    question: str,
    answer: str,
    n: int = 1,
    max_retries: int = 3,
    llm_type = LLMType(LLM_VALIDATOR),
):
    """
    Validate a question-answer pair using an LLM. 
    Each evaluation may be retried up to `max_retries` times if errors occur.
    
    Args:
        question (str): The input question.
        answer (str): The provided answer.
        n (int): Number of evaluations (≥1).
        max_retries (int): Maximum number of retries per evaluation.
        llm_type: Type of LLM to use (defaults to LLM_VALIDATOR).

    Returns:
        float: The average score across evaluations (rounded to 3 decimals).
    """
    if llm_type is None:
        llm_type = LLMType(LLM_VALIDATOR)
    assert n >= 1

    answers = []
    prompt_eval = PROMT_EVALUATE_REQUEST_RESPONSE_HELP.format(question, answer)

    for _ in range(n):
        attempt = 0
        score = None

        while attempt < max_retries:
            try:
                if not DEBUG and (llm_type != LLMType.MOCK):
                    raw_answer = pass_llm(
                        prompt_eval,
                        temperature=0.1,  # use fixed temperature
                        llm_type=llm_type,
                    )
                    repaired = repair_json(raw_answer)
                    parsed = json.loads(repaired)
                    score = float(parsed["score"])
                else:
                    score = float(random.random())
                break  # success, exit retry loop

            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"[llm_validator] Failed after {max_retries} attempts: {e}"
                    )

        answers.append(score)
    return round(np.mean(answers), 3)


def llm_output_validator(raw_output, n=1, llm_type = LLMType(LLM_VALIDATOR)):
    if llm_type is None:
        llm_type = LLMType(LLM_VALIDATOR)
    assert n >= 1
    answers = []
    prompt_eval = VALIDATION_PROMPT_RAW_OUTPUT.format(raw_output)
    for t in np.linspace(0, 1, n):
        if not DEBUG and (llm_type != LLMType.MOCK):
            answer = pass_llm(prompt_eval, 
                              temperature=t, 
                              llm_type=llm_type)
        else:
            answer = str(random.random()) + " \n"
        answers.append(float(answer.strip()))
    return round(np.mean(answers),3)

if __name__ == "__main__":
    from datetime import datetime
    import csv

    # Your evaluation examples
    questions = {
        "I am hungry": [
            "The weather is good.",
            "It is bad to be hungry.",
            "A restaurant could serve food to be not hungry anymore.",
            "You should look for a restaurant.",
            "Do you want me to look for an Italian restaurant nearby?"
        ],
        "I need new clothes.": [
            "Time for some drink.",
            "You surely need them.",
            "There are some malls with clothes",
            "You should go to a mall.",
            "Do you want me to navigate to a mall nearby?"
        ],
        "I feel cold.": [
            "I dont know what you want.",
            "It is good to freeze.",
            "It would make sense to wear something warm.",
            "You might want to check the thermostat.",
            "Would you like me to find a nearby store for warm clothes, there is one next to us?"
        ],
        "I am running out of gas" : [
            "Sounds funny.",
            "It is not good to run out of gas.",
            "You should go to some place where is gas.",
            "You should immediately approach a gas station.",
            "Do you want me to looks for some nearby and cheap gas station?"
        ],
        "I need some break." : [
            "It is a nice day today.",
            "It is time to rest for all of us.",
            "You should rest.",
            "You should look for some quiet place or hotel to rest.",
            "Do you want me to navigate to a park, hotel or wellness center?"
        ],
        "I am looking for a park." : [
            "Sorry, I did not understand you.",
            "I understand you need a park",
            "There are a lot parks worth to visit nearby.",
            "Do you want me to navigate to some park?",
            "Do you want me to navigate to the central park close by? This park has the best reviews."
        ]
    }

    models = ["gpt-35-turbo-1106", "gpt-4o", "llama3.2"]
    gt_score = [0, 0.25, 0.5, 0.75, 1]

    # Output CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"nuanced_validation_results_{timestamp}.csv"

    # Write to CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Question", "Answer", "Scores"])

        for model in models:
            for question, answer_list in list(questions.items())[-1:]:
                for answer in answer_list:
                    res = llm_validator(question, 
                                        answer, 
                                        llm_type=model, 
                                        n=1)
                    print(f"Model: {model} | Q: {question} | A: {answer} | Scores: {res}")
                    writer.writerow([model, question, answer, res])

    print(f"\nResults saved to {csv_file}")
