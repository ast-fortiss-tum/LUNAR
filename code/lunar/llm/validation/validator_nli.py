import os
from pathlib import Path
import random
from transformers import pipeline
import numpy as np
from llm.config import DEBUG
import torch
from llm.prompts import VALIDATION_PROMPT
import time
import numpy as np
import warnings
from transformers.utils import logging
import csv
import logging as log
from llm.config import N_VALIDATORS
from llm.prompts import SYSTEM_PROMPT
import traceback
from llm.utils import seed

seed.set_seed(43)

torch.cuda.empty_cache()

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

threshold = 0.7

############
def apply_nli(question, answer, model = "roberta-large-mnli", device = -1, mode = "entail-bi"): # contra, entail, entail-bi

    def prepr_question(text):
        return f"The answer is {text}"
    def prepr_answer(text):
        return f"The answer is: {text}"
    # alternative models: facebook/bart-large-mnli 
    nli_pipeline= pipeline("text-classification", model= model, device = device)

    if mode == "contra":
        result = nli_pipeline(f"{prepr_question(question)} </s> {prepr_answer(answer)}", return_all_scores=True)
        scores = {x["label"].lower(): x["score"] for x in result[0]}
        # log.info("[NLI] Question:", question)
        # log.info("[NLI] Answer:", answer)
        # log.info(scores)
        return 1 - scores["contradiction"]  
    elif mode == "entail":
        result = nli_pipeline(f"{prepr_question(question)} </s> {prepr_answer(answer)}", return_all_scores=True)
        scores_1 = {x["label"].lower(): x["score"] for x in result[0]}
        entail_1 = scores_1["entailment"]  
        return entail_1
    elif mode == "entail-bi":
        result = nli_pipeline(f"{prepr_question(question)} </s> {prepr_answer(answer)}", return_all_scores=True)
        scores_1 = {x["label"].lower(): x["score"] for x in result[0]}
        entail_1 = scores_1["entailment"]  
        
        result = nli_pipeline(f"{prepr_answer(answer)} </s> {prepr_question(question)}", return_all_scores=True)
        scores_2 = {x["label"].lower(): x["score"] for x in result[0]}
        entail_2 = scores_2["entailment"]  
        return (entail_1 + entail_2)/2
    else:
        raise ValueError("Unknown mode")

def llm_validator(question, answer, n = N_VALIDATORS, device = -1):
    answers = []
    for t in np.linspace(0, 1, n):
        if not DEBUG:
            answer = apply_nli(question, answer, device=device)
        else:
            answer = float(random.random())
        answers.append(answer)
    # print(answers)   
    return round(np.mean(answers),3)

def test_validator_nli(datasets, n_tests = None, device = -1):
    for dataset in datasets:
        
        ###################
        hypotheses = []
        premises = []
        gt_scores = []

        # take only the positive cases
        data_name = dataset["name"]
        gt_scores = dataset["gt_scores"]
        hypotheses = dataset["hypotheses"]
        premises = dataset["premises"]

        assert len(gt_scores) == len(hypotheses)
        assert len(hypotheses) == len(premises)

        def evaluate_nli(use_ensemble=True, n_ensemble = 3, mode="average"):
            # Load the NLI model
            nli_pipeline_1= pipeline("text-classification", model="facebook/bart-large-mnli", device = device)
            nli_pipeline_2 = pipeline("text-classification", model="cross-encoder/nli-deberta-base", device = device)
            nli_pipeline_3 = pipeline("zero-shot-classification", model="xlnet-large-cased", device = device)

            print("Evaluating...")
            
            # Store start time to measure total time
            start_time = time.time()
            
            all_scores = {"entailment": 0, "contradiction": 0, "neutral": 0}
            all_certainties = {"entailment": 0, "contradiction": 0, "neutral": 0}
            all_scores_detail = []

            total_inference_time = 0  # To accumulate total inference time
            
            # Perform inference
            for premise, hypothesis in list(zip(premises, hypotheses))[:n_tests]:
                scores = {"entailment": 0, 
                          "contradiction": 0, 
                          "neutral": 0}

                # Start the inference for each iteration
                inference_start_time = time.time()
                
                result_1 = nli_pipeline_1(f"{premise} </s> {hypothesis}", return_all_scores=True)
                
                if use_ensemble:
                    result_2 = nli_pipeline_2(f"{premise} </s> {hypothesis}", return_all_scores=True)
                    result_3 = nli_pipeline_3(sequences=premise, candidate_labels=["entailment", "contradiction", "neutral"], return_all_score = True)
                    if n_ensemble == 3:
                        scores_3 = {l: s for l,s in zip(result_3["labels"], result_3["scores"])}
                    else:
                        scores_3 = scores
                else:
                    result_2 = result_1
                    
                # Extract and format scores
                scores_1 = {x["label"]: x["score"] for x in result_1[0]}
                scores_2 = {x["label"]: x["score"] for x in result_2[0]}

                # Mode selection for score aggregation
                if mode == "average" and use_ensemble:
                    scores["contradiction"] = (scores_1["contradiction"] + scores_2["contradiction"] + scores_3["contradiction"]) / n_ensemble
                    scores["entailment"] = (scores_1["entailment"] + scores_2["entailment"] + scores_2["entailment"]) / n_ensemble
                    scores["neutral"] = (scores_1["neutral"] + scores_2["neutral"] + scores_3["neutral"]) / n_ensemble
                else:
                    max_contra = max(scores_1["contradiction"], scores_2["contradiction"])
                    if max_contra > 0.5:
                        scores["contradiction"] = max_contra
                        if np.argmax([scores_1["contradiction"], scores_2["contradiction"]]) == 0:
                            scores["entailment"] = scores_1["entailment"]
                            scores["neutral"] = scores_1["neutral"]
                        else:
                            scores["entailment"] = scores_2["entailment"]
                            scores["neutral"] = scores_2["neutral"]
                    else:
                        scores = scores_2

                # Store the result
                category = list(scores.keys())[np.argmax(list(scores.values()))]
                #value = np.max(np.round(list(scores.values()), 2))
                
                # Update overall score count
                all_scores[category] += 1
                for k,v in scores.items():
                    all_certainties[k] += v

                all_scores_detail.append(scores)
                    
                # Measure time for current iteration and accumulate
                inference_time = time.time() - inference_start_time
                total_inference_time += inference_time

            # Calculate average inference time
            average_inference_time = total_inference_time / len(premises)
            
            # Measure total time taken for the evaluation process
            total_time = time.time() - start_time
            print(f"Total evaluation time: {total_time:.4f} seconds")
            print(f"Average inference time per pair: {average_inference_time:.4f} seconds")
            all_certainties_ = [(key,float(v)/len(premises)) for key,v in all_certainties.items()]
            return all_scores, all_certainties_, all_scores_detail

        def evaluate_cosim(threshold):
            from llm.utils.embeddings_openai import get_similarity

            all_scores = {"entailment": 0, "contradiction": 0, "neutral": 0}
            all_certainties = {"entailment": 0, "contradiction": 0, "neutral": 0}
            all_scores_detail = []

            start_time = time.time()
            total_inference_time = 0  # To accumulate total inference time
            average_inference_time = 0
            inference_start_time = 0
            
            for premise, hypothesis in zip(premises, hypotheses):
                # Start the inference for each iteration
                inference_start_time = time.time()
                # print("premise", premise)
                # print("hypothesis", hypothesis)
                score = get_similarity(premise,hypothesis)
                #print(score)
                all_certainties["contradiction"] += 1 - abs((score+1))/2
                all_certainties["neutral"] += abs((score+1))/2
                all_certainties["entailment"] += 0
                
                all_scores["contradiction"] += 1 if abs((score+1))/2 < threshold else 0
                all_scores["neutral"] += 1 if abs((score+1))/2 >= threshold else 0
                all_scores["entailment"] += 0

                score_trace = {"contradiction" : 1 - abs((score+1))/2,
                                        "neutral" : abs((score+1))/2,
                                        "entailment" : 0}
                all_scores_detail.append(score_trace)
                # Measure time for current iteration and accumulate
                inference_time = time.time() - inference_start_time
                total_inference_time += inference_time

            # Calculate average inference time
            average_inference_time = total_inference_time / len(premises)
            
            # Measure total time taken for the evaluation process
            total_time = time.time() - start_time
            print(f"Total evaluation time: {total_time:.4f} seconds")
            print(f"Average inference time per pair: {average_inference_time:.4f} seconds")
            all_certainties_ = [(key,float(v)/ len(premises)) for key,v in all_certainties.items()]

            return all_scores, all_certainties_, all_scores_detail

        def evaluate_llm(prompt_template, repeat = 1):
            from llm.llms import pass_llm

            all_scores = {"entailment": 0, "contradiction": 0, "neutral": 0}
            all_certainties = {"entailment": 0, "contradiction": 0, "neutral": 0}
            all_scores_detail = []

            start_time = time.time()
            total_inference_time = 0  # To accumulate total inference time
            average_inference_time = 0
            inference_start_time = 0
            
            for premise, hypothesis in zip(premises, hypotheses):
                # Start the inference for each iteration
                inference_start_time = time.time()
                max_retry = 5
                retry = 0
                do_retry = True
                while do_retry and retry < max_retry:
                    try:
                        scores_llms = []
                        for i in range(repeat):
                            score_i = float(pass_llm(prompt=prompt_template.format(premise, hypothesis),
                                                        system_message="You are an intelligent question answer evaluator."))
                            scores_llms.append(score_i)
                            # print("Premise: ", premise)
                            # print("Hypothesis: ", hypothesis)
                            # print("score from llm", score)
                        score = sum(scores_llms)/repeat
                        break
                    except Exception as e:
                        retry = retry + 1
                        do_retry = True
                        print("Exception occurred retrieving llm score.")
                        traceback.print_exc()
                #print(score)
                all_certainties["contradiction"] += 1 - score
                all_certainties["neutral"] += score 
                all_certainties["entailment"] += 0
                
                all_scores["contradiction"] += 1 if score  < threshold else 0
                all_scores["neutral"] += 1 if score >= threshold else 0
                all_scores["entailment"] += 0

                score_trace = {"contradiction" : 1 - score,
                                        "neutral" : score,
                                        "entailment" : 0}
                all_scores_detail.append(score_trace)
                # Measure time for current iteration and accumulate
                inference_time = time.time() - inference_start_time
                total_inference_time += inference_time

            # Calculate average inference time
            average_inference_time = total_inference_time / len(premises)
            
            # Measure total time taken for the evaluation process
            total_time = time.time() - start_time
            print(f"Total evaluation time: {total_time:.4f} seconds")
            print(f"Average inference time per pair: {average_inference_time:.4f} seconds")
            all_certainties_ = [(key,float(v)/ len(premises)) for key,v in all_certainties.items()]

            return all_scores, all_certainties_, all_scores_detail

        def get_accuracy(all_scores_detail, gt_scores, threshold):
            acc = 0
            for i in range(len(gt_scores)):
                contradict= all_scores_detail[i]["contradiction"] > threshold
                acc += contradict != int(gt_scores[i])
            #print(acc)
            return acc/len(gt_scores)


        ################

        print("Number positive: ", sum(gt_scores))
        print("Number negative: ", len(gt_scores) - sum(gt_scores))

        # Run evaluation with ensemble and average mode
        all_scores_ensemble, all_certainties_ensemble, all_scores_detail_ensemble = evaluate_nli(use_ensemble=True, n_ensemble=3, mode="average")
        accuracy_ensemble = get_accuracy(all_scores_detail_ensemble, gt_scores, threshold=threshold)
        accuracy_ensemble_05 = get_accuracy(all_scores_detail_ensemble, gt_scores, threshold=0.5)

        # Run evaluation with ensemble and average mode
        print("################### Evaluation Mode: Ensemble, Average")
        print("Scores:", all_scores_ensemble)
        print("Accuracy:", accuracy_ensemble)
        print("Certainties:", all_certainties_ensemble)

        # Run evaluation without ensemble
        all_scores_no_ensemble, all_certainties_no_ensemble, all_scores_detail_no_ensemble = evaluate_nli(use_ensemble=False)
        accuracy_no_ensemble = get_accuracy(all_scores_detail_no_ensemble, gt_scores, threshold=threshold)
        accuracy_no_ensemble_05 = get_accuracy(all_scores_detail_no_ensemble, gt_scores, threshold=0.5)

        print("################### Evaluation Mode: NLI without Ensemble")
        print("Scores:", all_scores_no_ensemble)
        print("Accuracy:", accuracy_no_ensemble)
        print("Certainties:", all_certainties_no_ensemble)

        # Run evaluation using cosine similarity
        # Run evaluation using cosine similarity
        all_scores_cosim, all_certainties_cosim, all_scores_detail_cosim = evaluate_cosim(threshold=threshold)
        accuracy_cosim = get_accuracy(all_scores_detail_cosim, gt_scores, threshold=threshold)
        accuracy_cosim_05 = get_accuracy(all_scores_detail_cosim, gt_scores, threshold=0.5)

        print("################### Evaluation Mode: Cosine Similarity")
        print("Scores:", all_scores_cosim)
        print("Accuracy:", accuracy_cosim)
        print("Certainties:", all_certainties_cosim)

        llm_repeat_3 = 3
        all_scores_llm_3, all_certainties_llm_3, all_scores_detail_llm_3 = evaluate_llm(prompt_template=VALIDATION_PROMPT, repeat=llm_repeat_3)
        accuracy_llm_3 = get_accuracy(all_scores_detail_llm_3, gt_scores, threshold=threshold)
        accuracy_llm_05_3 = get_accuracy(all_scores_detail_llm_3, gt_scores, threshold=0.5)

        print("################### Evaluation Mode: LLM (repeat 3)")
        print("Scores:", all_scores_llm_3)
        print("Accuracy:", accuracy_llm_3)
        print("Certainties:", all_certainties_llm_3)

        # self consistency
        llm_repeat_1 = 1
        all_scores_llm_1, all_certainties_llm_1, all_scores_detail_llm_1 = evaluate_llm(prompt_template=VALIDATION_PROMPT, repeat=llm_repeat_1)
        accuracy_llm_1 = get_accuracy(all_scores_detail_llm_1, gt_scores, threshold=threshold)
        accuracy_llm_05_1 = get_accuracy(all_scores_detail_llm_1, gt_scores, threshold=0.5)

        print("################### Evaluation Mode: LLM (repeat 1)")
        print("Scores:", all_scores_llm_1)
        print("Accuracy:", accuracy_llm_1)
        print("Certainties:", all_certainties_llm_1)

        # Print evaluation results in table format
        eval_table = [
            ["NLI Ensemble", all_scores_ensemble, accuracy_ensemble, accuracy_ensemble_05,  all_certainties_ensemble],
            ["NLI Single", all_scores_no_ensemble, accuracy_no_ensemble, accuracy_no_ensemble_05, all_certainties_no_ensemble],
            ["Cosine Similarity", all_scores_cosim, accuracy_cosim, accuracy_cosim_05, all_certainties_cosim],
            [f"LLM (SC, {llm_repeat_1})", all_scores_llm_1, accuracy_llm_1, accuracy_llm_05_1, all_certainties_llm_1],
            [f"LLM (SC, {llm_repeat_3})", all_scores_llm_3, accuracy_llm_3, accuracy_llm_05_3, all_certainties_llm_3]
        ]
        pos = sum(gt_scores)
        neg = len(gt_scores) - pos
        headers = ["Evaluation Mode", f"Scores (pos/neg = {pos} / {neg})", f"Accuracy (>{threshold})", f"Accuracy (>{0.5})", "Certainties Average"]

        # Save evaluation results to a CSV file
        csv_output_path = f"./llm/out/evaluation_results_{data_name}.csv"
        Path(os.path.dirname(csv_output_path)).mkdir(exist_ok=True, parents=True)

        with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(headers)
            csv_writer.writerows(eval_table)

        print(f"Evaluation results saved to {csv_output_path}")

if __name__ == "__main__":
    from llm.test.validator_data import implicit_planning_dataset_1#, \
                                    # artifical_music_dataset, \
                                    # basic_dataset, \
                                    # implicit_planning_dataset_2, \
                                    # context_understanding_dataset, \
                                    # feedback_dataset, \
                                    # negation_dataset, \
                                    # pizza_dataset, \
                                    # industry_dataset
    datasets = [
                    # pizza_dataset,
                    # negation_dataset
                    implicit_planning_dataset_1#, 
                    # implicit_planning_dataset_2, 
                    # artifical_music_dataset, 
                    # context_understanding_dataset,
                    # basic_dataset,
                    # feedback_dataset,
                    # industry_dataset
    ]
    
    test_validator_nli(datasets=datasets)