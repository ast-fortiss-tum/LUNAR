import sys
from pathlib import Path
import site

# import here the lunar project and its dependencies as we are using its llm calling interface and
# cost tracking
import sys
sys.path.insert(0, "../lunar/venv/lib/python3.11/site-packages")
sys.path.insert(0, "../lunar/")

from importlib.resources import path
import time
import timeit
import json
from copy import deepcopy
from argparse import ArgumentParser
from datetime import datetime, timedelta
import random
import os

import wandb

from user_sim.utils.config import errors
import pandas as pd
import yaml
from colorama import Fore, Style
from technologies.chatbot_connectors import (
    Chatbot, ChatbotConvNavi
)
from user_sim.data_extraction import DataExtraction
from user_sim.role_structure import *
from user_sim.user_simulator import UserGeneration
from user_sim.utils.show_logs import *
from user_sim.utils.utilities import *
from eval.navi.adapter import convert_to_simout, evaluate_simout, save_simout, write_token_usage


def print_user(msg): print(f"{Fore.GREEN}User:{Style.RESET_ALL} {msg}")


def print_chatbot(msg): print(f"{Fore.LIGHTRED_EX}Chatbot:{Style.RESET_ALL} {msg}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _execute_with_input_compat(the_chatbot, user_msg, user_id=None, llm_type=None):
    """
    Compatibility wrapper:
      - Some connectors return (is_ok, response)
      - Newer ones return (is_ok, response, retrieved_obj)

    Returns: (is_ok, response, retrieved_obj_or_None)
    """
    out = the_chatbot.execute_with_input(user_msg, user_id=user_id, llm_type=llm_type)

    if isinstance(out, tuple) and len(out) == 3:
        is_ok, response, retrieved_obj = out
        return is_ok, response, retrieved_obj
    if isinstance(out, tuple) and len(out) == 2:
        is_ok, response = out
        return is_ok, response, None

    raise TypeError(
        f"execute_with_input returned unexpected value: {type(out)} {out!r}. "
        "Expected tuple (is_ok, response) or (is_ok, response, retrieved_obj)."
    )


def _execute_starter_chatbot_compat(the_chatbot):
    """
    Compatibility wrapper for optional execute_starter_chatbot().
    Returns: (is_ok, response, retrieved_obj_or_None)
    """
    out = the_chatbot.execute_starter_chatbot()

    if isinstance(out, tuple) and len(out) == 3:
        is_ok, response, retrieved_obj = out
        return is_ok, response, retrieved_obj
    if isinstance(out, tuple) and len(out) == 2:
        is_ok, response = out
        return is_ok, response, None

    raise TypeError(
        f"execute_starter_chatbot returned unexpected value: {type(out)} {out!r}. "
        "Expected tuple (is_ok, response) or (is_ok, response, retrieved_obj)."
    )


def generate_problem_name(
    algo: str,
    sut: str,
    population_size: int,
    generator_llm: str,
    judge_llm: str,
    seed: int,
    max_time: str,
    sut_llm:str,
    personality_name: str,
) -> str:
    # <algo>_<sut>_n<population_size>_time<max_time>_gen<generator_llm>_judge<judge_llm>_seed<seed>_pers<personality>
    safe_time = str(max_time).replace(":", "-")
    return (
        f"{algo.upper()}_{sut.lower()}_{sut_llm}_{population_size}n_{safe_time}t_{seed}seed"
    )


def parse_max_time(max_time_raw):
    """
    Accepts:
      - None -> returns None
      - "None" (case-insensitive) -> returns None
      - number-like string (seconds) -> float seconds
      - float/int -> float seconds
      - "hh:mm:ss" -> float seconds
      - "mm:ss" -> float seconds
    """
    if max_time_raw is None:
        return None

    if isinstance(max_time_raw, (int, float)):
        if max_time_raw < 0:
            raise ValueError("--max_time must be >= 0")
        return float(max_time_raw)

    s = str(max_time_raw).strip()
    if s.lower() == "none":
        return None

    # Try plain seconds first: "90" or "90.5"
    try:
        seconds = float(s)
        if seconds < 0:
            raise ValueError("--max_time must be >= 0")
        return seconds
    except ValueError:
        pass

    # Try "mm:ss" or "hh:mm:ss"
    parts = s.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f'Invalid --max_time "{s}". Use seconds (e.g. "90"), "mm:ss", "hh:mm:ss", or "None".'
        )

    if len(parts) == 2:
        hh = 0
        mm, ss = parts
    else:
        hh, mm, ss = parts

    try:
        hh = int(str(hh).strip())
        mm = int(str(mm).strip())
        ss = float(str(ss).strip())  # allow fractional seconds

        if hh < 0 or mm < 0 or ss < 0:
            raise ValueError("--max_time must be >= 0")
        if mm >= 60 or ss >= 60:
            raise ValueError(f'Invalid --max_time "{s}": minutes/seconds must be < 60.')

        return hh * 3600 + mm * 60 + ss
    except ValueError as e:
        raise ValueError(
            f'Invalid --max_time "{s}". Use seconds, "mm:ss", "hh:mm:ss", or "None".'
        ) from e


def iter_personality_files(personality_path: str | None):
    """
    If personality_path is:
      - None: yield (None, "None")
      - a file: yield (that file, stem)
      - a directory: yield all *.yml/*.yaml files inside (sorted)
    """
    if not personality_path:
        yield None, "None"
        return

    p = Path(personality_path)
    if p.is_file():
        yield str(p), p.stem
        return

    if p.is_dir():
        files = sorted([*p.rglob("*.yml"), *p.rglob("*.yaml")], key=lambda x: str(x))
        if not files:
            raise ValueError(f"No .yml/.yaml files found in personality folder: {personality_path}")
        for f in files:
            yield str(f), f.stem
        return

    raise ValueError(f"Invalid --personality path: {personality_path}")


# -----------------------------
# W&B aggregate logging helper
# -----------------------------
class WandbAggregateLogger:
    """
    Mirrors opensbt logging_callback_archive:
      - test_size
      - failures
      - critical_ratio
      - timestamp
    """
    def __init__(self):
        self.test_size = 0
        self.failures = 0

    def update(self, is_failure: bool):
        self.test_size += 1
        if is_failure:
            self.failures += 1

        wandb.log(
            {
                "test_size": self.test_size,
                "failures": self.failures,
                "critical_ratio": (self.failures / self.test_size) if self.test_size > 0 else 0.0,
                "timestamp": time.time(),
            },
            step=self.test_size,
        )


def get_conversation_metadata(user_profile, the_user, serial=None):
    def conversation_metadata(up):
        interaction_style_list = []
        conversation_list = []

        for inter in up.interaction_styles:
            interaction_style_list.append(inter.get_metadata())

        conversation_list.append({'interaction_style': interaction_style_list})

        if isinstance(up.yaml['conversation']['number'], int):
            conversation_list.append({'number': up.yaml['conversation']['number']})
        else:
            conversation_list.append({'number': up.conversation_number})

        if 'random steps' in up.yaml['conversation']['goal_style']:
            conversation_list.append({'goal_style': {'steps': up.goal_style[1]}})
        else:
            conversation_list.append(up.yaml['conversation']['goal_style'])

        return conversation_list

    def ask_about_metadata(up):
        if not up.ask_about.variable_list:
            return up.ask_about.str_list
        return user_profile.ask_about.str_list + user_profile.ask_about.picked_elements

    def data_output_data_extraction(u_profile, user):
        output_list = u_profile.output
        data_list = []
        for output in output_list:
            var_name = list(output.keys())[0]
            var_dict = output.get(var_name)
            my_data_save_folder = DataExtraction(
                user.conversation_history,
                var_name,
                var_dict["type"],
                var_dict["description"]
            )
            data_list.append(my_data_save_folder.get_data_extraction())
        print("data_list: ", data_list)

        data_dict = {k: v for dic in data_list for k, v in dic.items()}
        has_none = any(value is None for value in data_dict.values())
        if has_none:
            count_none = sum(1 for value in data_dict.values() if value is None)
            errors.append({1001: f"{count_none} goals left to complete."})

        return data_list

    data_output = {'data_output': data_output_data_extraction(user_profile, the_user)}
    context = {'context': user_profile.raw_context}
    ask_about = {'ask_about': ask_about_metadata(user_profile)}
    conversation = {'conversation': conversation_metadata(user_profile)}
    language = {'language': user_profile.language}
    serial_dict = {'serial': serial}
    errors_dict = {'errors': errors}
    variables_per_turn = {'variables_per_turn': the_user.variables_per_turn}

    metadata = {
        **serial_dict,
        **language,
        **context,
        **ask_about,
        **conversation,
        **data_output,
        **errors_dict,
        **variables_per_turn,
    }

    return metadata


def parse_profiles(user_path):
    def is_yaml(file):
        if not file.endswith(('.yaml', '.yml')):
            return False
        try:
            with open(file, 'r') as f:
                yaml.safe_load(f)
            return True
        except yaml.YAMLError:
            return False

    list_of_files = []
    if os.path.isfile(user_path):
        if is_yaml(user_path):
            yaml_file = read_yaml(user_path)
            return [yaml_file]
        else:
            raise Exception(f'The user profile file is not a yaml: {user_path}')
    elif os.path.isdir(user_path):
        for root, _, files in os.walk(user_path):
            for file in files:
                if is_yaml(os.path.join(root, file)):
                    pth = root + '/' + file
                    yaml_file = read_yaml(pth)
                    list_of_files.append(yaml_file)
            return list_of_files
    else:
        raise Exception(f'Invalid path for user profile operation: {user_path}')


def build_chatbot(technology, chatbot) -> Chatbot:
    default = Chatbot
    chatbot_builder = {
        'convnavi': ChatbotConvNavi,
    }
    if technology in chatbot_builder:
        return chatbot_builder[technology](chatbot)
    else:
        return default(chatbot)


def build_summary_metadata_from_args(args, execution_time_seconds: float, actual_conversations_completed: int) -> dict:
    timestamp = datetime.now().isoformat()

    return {
        "seed": args.seed,
        "algorithm": args.algorithm,
        "population_size": args.population_size,
        "generations": None,
        "sut": args.sut,
        "sut_llm": args.sut_llm,
        "weight_clarity": args.weight_clarity,
        "weight_request_orientedness": args.weight_request_orientedness,
        "critical_threshold": args.critical_threshold,
        "execution_time_seconds": execution_time_seconds,
        "timestamp": timestamp,
        "actual_generations_completed": actual_conversations_completed,
        "generator_llm": args.generator_llm,
        "judge_llm": args.judge_llm,
        "max_time": args.max_time,
        "personality": getattr(args, "_resolved_personality_file", None),
        "personality_name": getattr(args, "_resolved_personality_name", None)
    }


def generate(technology, chatbot, user, personality, save_folder, summary_args, total_start=None, sut_llm="gpt-5-chat", generator_llm="gpt-4o"):
    """
    Runs conversations for a SINGLE personality file.

    Important: max_time is GLOBAL across personalities if total_start is provided (same timer reference).
    """
    set_global_seed(summary_args.seed)

    print("selected sut llm:", sut_llm)

    max_time_seconds = parse_max_time(summary_args.max_time)
    if total_start is None:
        total_start = timeit.default_timer()

    profiles = parse_profiles(user)
    serial = generate_serial()
    my_execution_stat = ExecutionStats(save_folder, serial)

    agg_logger = WandbAggregateLogger()

    all_evaluated_conversations = []
    total_conversations_completed = 0

    for profile in profiles:
        user_profile = RoleData(profile, personality)
        test_name = user_profile.test_name
        start_time_test = timeit.default_timer()

        conv_cap = summary_args.population_size if max_time_seconds is None else None

        print("Convcap limit:", conv_cap)
        print("conversation_number (profile): ", user_profile.conversation_number)
        if conv_cap is not None:
            print("conversation cap (args_population size): ", conv_cap)
        else:
            print("time budget (seconds): ", max_time_seconds)

        i = 0
        while True:
            # GLOBAL budget stop condition (shared across personalities if total_start shared)
            if max_time_seconds is not None:
                elapsed = timeit.default_timer() - total_start
                if elapsed >= max_time_seconds:
                    print(f"Max time budget reached: elapsed={elapsed:.3f}s >= budget={max_time_seconds:.3f}s")
                    break

            if conv_cap is not None and i >= conv_cap:
                break

            the_chatbot = build_chatbot(technology, chatbot)
            the_chatbot.fallback = user_profile.fallback    

            # helper defined once (outside the while loop)
            def id_from_ns(seed=0):
                ns = time.time_ns()  # current time in nanoseconds
                # keep it in a bounded range (0..999999) similar to your logic
                return ((ns // 800) + seed * 2) % 1_000_000
            user_id = id_from_ns()  # generate user_id once per conversation, can be used in all turns
            print("User ID:", user_id)

            # IMPORTANT: no intent-era llm_generator arg
            the_user = UserGeneration(user_profile, the_chatbot, user_id=user_id)

            bot_starter = user_profile.is_starter
            response_time = []

            start_time_conversation = timeit.default_timer()
            response = ""

            while True:
                if not bot_starter:
                    # user starts
                    user_msg = the_user.open_conversation()
                    print_user(user_msg)

                    start_response_time = timeit.default_timer()
                    is_ok, response, retrieved_obj = _execute_with_input_compat(
                        the_chatbot,
                        user_msg,
                        user_id=getattr(the_user, "user_id", None),
                        llm_type=sut_llm,
                    )
                    if retrieved_obj is not None and hasattr(the_user, "retrieved_objs_per_turn"):
                        the_user.retrieved_objs_per_turn.append(retrieved_obj)
                    end_response_time = timeit.default_timer()
                    response_time.append(timedelta(seconds=end_response_time - start_response_time).total_seconds())

                    if not is_ok:
                        the_user.update_history(
                            "Assistant",
                            ("Error: " + response) if response is not None else "Error: The server did not respond.",
                        )
                        break
                    else:
                        the_user.update_history(
                            "Assistant",
                            response if response is not None else "Error: The server did not respond.",
                        )

                    print_chatbot(response)
                    bot_starter = True
                    continue

                # chatbot starts (old behavior) if supported
                if bot_starter and not the_user.conversation_history["interaction"]:
                    if hasattr(the_chatbot, "execute_starter_chatbot"):
                        is_ok, response, retrieved_obj = _execute_starter_chatbot_compat(the_chatbot)
                        if retrieved_obj is not None and hasattr(the_user, "retrieved_objs_per_turn"):
                            the_user.retrieved_objs_per_turn.append(retrieved_obj)

                        if not is_ok:
                            the_user.update_history(
                                "Assistant",
                                ("Error: " + response) if response is not None else "Error: The server did not respond.",
                            )
                            break

                        print_chatbot(response)
                        user_msg = the_user.open_conversation(response)
                    else:
                        user_msg = the_user.open_conversation()
                else:
                    # IMPORTANT: old method name
                    user_msg = the_user.get_response(response,
                                    llm_type=generator_llm)

                if user_msg == "exit":
                    # if the_user.goal_style[2] > the_user.interaction_count:
                    #     user_msg = "Start navigation."
                    #     the_user.update_history("User", user_msg)
                    #     # send to sut
                    #     print_user(user_msg)
                    #     is_ok, response, retrieved_obj = _execute_with_input_compat(
                    #         the_chatbot,
                    #         user_msg,
                    #         user_id=getattr(the_user, "user_id", None),
                    #         llm_type=sut_llm,
                    #     )
                    #     the_user.update_history("Assistant", response)
                    #     print_chatbot(response)

                    #     the_user.interaction_count += 1

                    #     # optional bookkeeping, no intent logic
                    #     the_user.variables_per_turn.append({})
                    #     the_user.phrases_per_turn.append(None)
                    break

                print_user(user_msg)

                print("Getting response from the chatbot...")
                start_response_time = timeit.default_timer()
                is_ok, response, retrieved_obj = _execute_with_input_compat(
                    the_chatbot,
                    user_msg,
                    user_id=getattr(the_user, "user_id", None),
                    llm_type=sut_llm,
                )
                if retrieved_obj is not None and hasattr(the_user, "retrieved_objs_per_turn"):
                    the_user.retrieved_objs_per_turn.append(retrieved_obj)
                end_response_time = timeit.default_timer()

                response_time.append(timedelta(seconds=end_response_time - start_response_time).total_seconds())

                if response == "timeout":
                    break

                print_chatbot(response)

                if not is_ok:
                    the_user.update_history(
                        "Assistant",
                        ("Error: " + response) if response is not None else "Error: The server did not respond.",
                    )
                    break
                else:
                    the_user.update_history("Assistant", response)

            end_time_conversation = timeit.default_timer()
            conversation_time = end_time_conversation - start_time_conversation
            formatted_time_conv = timedelta(seconds=conversation_time).total_seconds()
            print(f"Conversation Time: {formatted_time_conv} (s)")

            simout = convert_to_simout(user_profile, the_user, serial, time_conv=formatted_time_conv)
            eval_result = evaluate_simout(simout, args=summary_args)
            simout_dict = simout.to_dict()
            simout_dict.update(eval_result)
            all_evaluated_conversations.append(simout_dict)

            is_failure = bool(eval_result.get("is_critical", False))
            agg_logger.update(is_failure=is_failure)

            fitness_scores = eval_result.get("fitness_scores", {})
            if isinstance(fitness_scores, dict) and fitness_scores:
                wandb.log({f"fitness/{k}": v for k, v in fitness_scores.items()}, step=agg_logger.test_size)

            total_conversations_completed += 1

            ############### ORIGINAL OUTPUT CODE #######
            end_time_conversation = timeit.default_timer()
            conversation_time = end_time_conversation - start_time_conversation
            formatted_time_conv = timedelta(seconds=conversation_time).total_seconds()
            print(f"Conversation Time: {formatted_time_conv} (s)")

            history = the_user.conversation_history
            metadata = get_conversation_metadata(user_profile, the_user, serial)
            dg_dataframe = the_user.data_gathering.gathering_register
            csv_extraction = the_user.goal_style[1] if the_user.goal_style[0] == 'all_answered' else False
            answer_validation_data = (dg_dataframe, csv_extraction)
            save_test_conv(history, metadata, test_name, save_folder, serial,
                            formatted_time_conv, response_time, answer_validation_data, counter=i)

            ######################

            user_profile.reset_attributes()
            i += 1

        end_time_test = timeit.default_timer()
        execution_time = end_time_test - start_time_test
        formatted_time = timedelta(seconds=execution_time).total_seconds()
        print(f"Execution Time: {formatted_time} (s)")
        print("------------------------------")

        if max_time_seconds is not None:
            elapsed = timeit.default_timer() - total_start
            if elapsed >= max_time_seconds:
                break

    total_end = timeit.default_timer()
    total_execution_seconds = timedelta(seconds=(total_end - total_start)).total_seconds()

    path_folder = save_folder + f"/{test_name}" + f"/{serial}"
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    out_path = path_folder + f"/report.json"

    write_token_usage(save_folder=path_folder)

    summary_metadata = build_summary_metadata_from_args(
        summary_args,
        execution_time_seconds=total_execution_seconds,
        actual_conversations_completed=total_conversations_completed,
    )

    wandb.summary.update(summary_metadata)
    wandb.summary["total_conversations_completed"] = total_conversations_completed

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": summary_metadata, "all_evaluated_conversations": all_evaluated_conversations},
            f,
            ensure_ascii=False,
            indent=2,
    )
    
    try:
        artifact = wandb.Artifact("results_folder", "output")
        artifact.add_dir(path_folder)
        wandb.log_artifact(artifact)
    except Exception as e:
        print("Error happened when uploading to wandb.")
        print(e)

    print(f"Saved {len(all_evaluated_conversations)} evaluated conversations to {out_path}")

    # Return elapsed so caller can stop before starting next personality
    if max_time_seconds is None:
        return False  # "not timed out"
    return (timeit.default_timer() - total_start) >= max_time_seconds


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Conversation generator for a chatbot")

    parser.add_argument(
        "--technology",
        required=True,
        choices=[
            "industry",
            "convnavi",
            "rasa",
            "taskyto",
            "ada-uam",
            "millionbot",
            "genion",
            "lola",
            "serviceform",
            "kuki",
            "julie",
            "rivas_catalina",
            "saic_malaga",
        ],
        help="Technology the chatbot is implemented in",
    )
    parser.add_argument("--chatbot", required=True, help="URL where the chatbot is deployed")
    parser.add_argument("--user", required=True, help="User profile to test the chatbot")
    parser.add_argument("--personality", required=False, help="Personality file OR folder of personality yaml files")
    parser.add_argument("--save_folder", default=False, help="Path to store conversation user-chatbot")

    parser.add_argument("--seed", type=int, default=1, help="Random seed used for the run")
    parser.add_argument("--algorithm", type=str, default="sensei", help="Algorithm name (e.g., rs)")
    parser.add_argument("--population_size", dest="population_size", type=int, required=True, help="Population size")
    parser.add_argument("--sut", type=str, default="IPA_YELP", help="System under test (e.g., ipa_yelp)")

    parser.add_argument("--weight_clarity", dest="weight_clarity", type=float, default=0.5, help="Weight for clarity metric")
    parser.add_argument(
        "--weight_request_orientedness",
        dest="weight_request_orientedness",
        type=float,
        default=0.5,
        help="Weight for request-orientedness metric",
    )   
    parser.add_argument(
        "--max_time",
        dest="max_time",
        type=str,
        default="None",
        help='GLOBAL time budget for the whole run (across personalities) in "hh:mm:ss", or "None"',
    )
    parser.add_argument("--critical_threshold", dest="critical_threshold", type=float, default=0.65)

    parser.add_argument("--generator_llm", dest="generator_llm", type=str, required=True, help="Generator LLM name/id used in problem_name")
    parser.add_argument("--judge_llm", dest="judge_llm", type=str, required=True, help="Judge LLM name/id used in problem_name")
    parser.add_argument("--sut_llm", dest="sut_llm", default="gpt-4o", help="SUT LLM name/id used in problem_name")

    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="NaviYelp", help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default="mt-test", help="Weights & Biases entity (team/user).")
    parser.add_argument("--wandb_group", type=str, default=None, help="Optional W&B group (defaults to run date).")
    parser.add_argument(
        "--shuffle_personalities",
        action="store_true",
        help="Shuffle personality files before running. Max time budget is still global across all personalities.",
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    logger = create_logger(True, "Info Logger")
    logger.info("Logs enabled!")

    check_keys(["OPENAI_API_KEY"])

    total_start_global = timeit.default_timer()
    max_time_seconds = parse_max_time(args.max_time)

    personalities = list(iter_personality_files(args.personality))
    if args.shuffle_personalities:
        random.shuffle(personalities)

    for personality_file, personality_name in personalities:
        # Stop BEFORE starting a new personality if global time budget is exceeded
        if max_time_seconds is not None:
            elapsed = timeit.default_timer() - total_start_global
            print("elapsed time is:", elapsed)
            if elapsed >= max_time_seconds:
                print(f"[GLOBAL STOP] Max time reached before starting next personality: {elapsed:.3f}s >= {max_time_seconds:.3f}s")
                break

        args._resolved_personality_file = personality_file
        args._resolved_personality_name = personality_name

        run_name = generate_problem_name(
            algo=args.algorithm,
            sut=args.sut, # need to fix
            population_size=args.population_size,
            generator_llm=args.generator_llm,
            judge_llm=args.judge_llm,
            seed=args.seed,
            max_time=args.max_time,
            personality_name=personality_name,
            sut_llm=args.sut_llm
        )

        tags = [f"{k}:{v}" for k, v in vars(args).items() if not k.startswith("_")]

        if args.no_wandb:
            wandb.init(mode="disabled")
        else:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=run_name,
                group=args.wandb_group or datetime.now().strftime("%d-%m-%Y"),
                tags=tags,
                config={
                    **{k: v for k, v in vars(args).items() if not k.startswith("_")},
                    "personality_file": personality_file,
                    "personality_name": personality_name,
                },
            )

        try:
            timed_out = generate(
                args.technology,
                args.chatbot,
                args.user,
                personality_file,
                args.save_folder,
                summary_args=args,
                total_start=total_start_global,
                sut_llm=args.sut_llm,  # <-- shared clock => GLOBAL max_time across personalities
                generator_llm=args.generator_llm,
            )
        finally:
            wandb.finish()

        # If we hit the global budget during this personality, stop the outer loop too
        if max_time_seconds is not None:
            elapsed = timeit.default_timer() - total_start_global
            if elapsed >= max_time_seconds:
                print(f"[GLOBAL STOP] Max time reached after personality run: {elapsed:.3f}s >= {max_time_seconds:.3f}s")
                break