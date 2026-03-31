"""
Generate conversations with controlled Clarity & Request-orientedness scores
using the IPA_LOS pipeline with SYSTEM_PROMPT_CONTENT_INPUT_HISTORY_DIMS prompt.

Produces:
  - 9 conversations covering all (clarity, req_orient) pairs in {0,1,2}x{0,1,2}
  - 6 conversations with random score pairs

Usage:
    python -m examples.navi.navi-survey.generate_convs_openai \
        --features_config configs/features_simple_judge_industry.json \
        --seed 42 \
        --min_turns 3 \
        --max_turns 6 \
        --max_repeats 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from matplotlib import text

from json_repair import repair_json

from llm.llms import LLMType, pass_llm
from llm.config import LLM_CLASSIFIER, LLM_IPA
from llm.features import FeatureHandler
from llm.model.models import Conversation, Turn, Utterance
from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.model.conversation_intents import UserIntent
from llm.sut.ipa_base_cc import IPABase
from llm.utils.seed import set_seed

from examples.car_control.models_new import CCContentInput, StyleDescription
from examples.car_control.cc_utterance_generator_new import CCUtteranceGenerator
from examples.car_control.cc_conversation_generator import CCConversationGenerator
from .cc_sampler import CCFeatureSampler, JUDGE_DIMENSIONS
from .llm_gen_prompt import GENERATION_PROMPT
from .prompts import SYSTEM_PROMPT_CONTENT_INPUT_HISTORY_DIMS_NEW_CARCONTROL


# ---------------------------------------------------------------------------
# IPA variant that injects dimension targets into the system prompt
# ---------------------------------------------------------------------------

class IPADims(IPABase):
    """
    IPA that uses SYSTEM_PROMPT_CONTENT_INPUT_HISTORY_DIMS with explicit
    clarity / request-orientedness target levels.  No `los` field is produced.
    """
    ipa_name = "openai_dims"
    global_user_counter = 0

    # Will be set before each conversation
    _clarity_level: int = 2
    _request_orientedness_level: int = 2

    @staticmethod
    def send_utterance_request(
        request: str,
        temperature: float = 0.3,
        context: object = None,
        system_message: str = "",
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        attempt = 0
        while attempt < max_retries:
            try:
                print("Sending request:", request)
                response = pass_llm(
                    msg=request,
                    llm_type=LLMType(LLM_IPA),
                    temperature=temperature,
                    context=context,
                    system_message=system_message,
                )

                if response is None or response.strip() == "":
                    print(f"[IPADims] Attempt {attempt + 1}: LLM returned empty response, retrying...")
                    attempt += 1
                    continue
                
                # print("type", type(response))
                # print("response", response)
                # Replace commas that are NOT immediately after punctuation (for incorrect filtering by json-repair)
                response = re.sub(r'(?<![.!?])\s*,', ' -', response)

                # response_parsed_["system_response"] = response_parsed_["system_response"].replace(",", "-")  # ✅ dict, string literal key
                # response = json.dumps(response_parsed_)

                # Try to extract JSON from the response
                repaired = repair_json(response)

                # If repair_json returns empty string, try to extract system_response manually
                if not repaired or repaired.strip() == "":
                    # The LLM might have returned plain text without JSON
                    print(f"[IPADims] repair_json returned empty, using raw response as system_response")
                    return {"data": {"result": response.strip()}}

                try:
                    response_parsed = json.loads(repaired)
                except json.JSONDecodeError:
                    # Still can't parse — use the raw response as plain text
                    print(f"[IPADims] JSON parse failed after repair, using raw response as system_response")
                    return {"data": {"result": response.strip()}}

                answer = response_parsed.get("system_response")
                if answer is None:
                    # Try common alternative keys
                    answer = response_parsed.get("response") or response_parsed.get("answer")
                if answer is None:
                    # Use the raw LLM text as the answer
                    answer = response.strip()
                
                answer = answer.replace(" -", ",")  # Replace back o avoid JSON issues

                return {"data": {"result": answer}}
            except Exception as e:
                traceback.print_exc()
                print(f"[IPADims] Attempt {attempt + 1} failed: {e}")
            attempt += 1

        return {"data": {"result": "Request after max retries failed."}}

    @staticmethod
    def simulate(
        list_individuals, variable_names, scenario_path, sim_time,
        time_step=10, do_visualize=False, temperature=0, context=None,
        max_retries=3,
    ):
        raise NotImplementedError("Single-turn simulate not used here.")

    @staticmethod
    def simulate_turn(
        user_text: str,
        user_intent: str,
        user_id: str,
        current_content_input: Optional[CCContentInput],
        history: List[str],
        max_retries: int = 3,
        **kwargs,
    ) -> Turn:
        context = kwargs.get("context")
        conversation: Optional[Conversation] = kwargs.get("conversation")

        effective_system_message = SYSTEM_PROMPT_CONTENT_INPUT_HISTORY_DIMS_NEW_CARCONTROL.format(
            context=context,
            history=conversation.get_dialogue_history_str() if conversation else "",
            clarity_level=IPADims._clarity_level,
            request_orientedness_level=IPADims._request_orientedness_level,
        )

        response_obj = IPADims.send_utterance_request(
            user_text,
            system_message=effective_system_message,
            max_retries=max_retries,
        )

        sys_answer_text = response_obj.get("data", {}).get("result", "System Error")
        print(f"[IPADims] System: {sys_answer_text}")
        print(f"###### system intents: {conversation.get_intent_history_dict()}")
        print("################")


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
        config_path: str = "configs/features_simple_judge_cc.json",
        max_retries: int = 3,
        min_turns: int = 2,
        max_turns: int = 5,
        max_repeats: int = 2,
    ) -> List[MultiTurnSimulationOutput]:

        feature_handler = FeatureHandler.from_json(config_path)
        utterance_gen = CCUtteranceGenerator(feature_handler=feature_handler)
        llm_ipa = LLMType(LLM_IPA)
        llm_classifier = LLMType(LLM_CLASSIFIER)

        results = []

        for conversation_wrapper in list_individuals:
            conversation = conversation_wrapper[0]

            user_id = f"user_{IPADims.global_user_counter}"
            IPADims.global_user_counter += 1
            conversation.assigned_user_id = user_id


            conversation = IPADims.run_conversation_loop(
                conversation=conversation,
                feature_handler=feature_handler,
                utterance_gen=utterance_gen,
                llm_ipa=llm_ipa,
                llm_classifier=llm_classifier,
                context=context,
                max_retries=max_retries,
                min_turns=min_turns,
                max_turns=max_turns,
                max_repeats=max_repeats,
            )

            result = MultiTurnSimulationOutput(
                conversation=conversation,
                model=llm_ipa,
                ipa=IPADims.ipa_name,
            )
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# Conversation-to-JSON serialisation (matching conv_0001.json format)
# ---------------------------------------------------------------------------

def conversation_to_json(
    conversation: Conversation,
    individual_id: int,
    clarity: int,
    request_orientedness: int,
    generation_time_seconds: float = 0.0,
    num_turns: int = 0,
) -> Dict[str, Any]:
    """Serialise a Conversation to the survey JSON format (no los)."""

    turns_list = []
    for t in conversation.turns:
        turns_list.append({
            "user": t.question,
            "system": t.answer,
        })

    # Extract content_input / style_input from the conversation metadata
    content_input_dict: Dict[str, Any] = {}
    style_input_dict: Dict[str, Any] = {}

    if conversation.turns:
        first_turn = conversation.turns[0]
        if first_turn.content_input is not None:
            content_input_dict = {
                k: v for k, v in first_turn.content_input.model_dump().items()
                if v is not None
            }

    # Try to recover style from feature_values on the conversation
    if hasattr(conversation, "categorical_vars") and hasattr(conversation, "ordinal_vars"):
        try:
            # We can't easily recover feature_handler here, so store raw vars
            pass
        except Exception:
            pass

    # Build fitness_scores placeholder
    fitness_scores: Dict[str, Any] = {}

    # Check criticality (placeholder)
    is_critical = False
    
    # Calculate average turn time
    avg_turn_time = generation_time_seconds / num_turns if num_turns > 0 else 0.0

    return {
        "generation_plan": (
            f"Generated via IPADims with SYSTEM_PROMPT_CONTENT_INPUT_HISTORY_DIMS. "
            f"Target Clarity={clarity}, Request-orientedness={request_orientedness}."
        ),
        "turns": turns_list,
        "raw_conversation": "",
        "metadata": {
            "individual_id": individual_id,
            "is_critical": is_critical,
            "judge_dimensions": {
                "Clarity": clarity,
                "Request-orientedness": request_orientedness,
            },
            "content_input": content_input_dict,
            "style_input": style_input_dict,
            "intent_history": conversation.get_intent_history_dict(),
            "fitness_scores": fitness_scores,
            "generation_time_seconds": round(generation_time_seconds, 2),
            "average_turn_time_seconds": round(avg_turn_time, 2),
        },
    }


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate_all_conversations(
    features_config: str,
    seed: int,
    min_turns: int,
    max_turns: int,
    max_repeats: int,
    output_dir: Optional[str] = None,
    num_conversations: int = 15,
    scores: List[int] = [0, 1, 2],
) -> Tuple[List[Path], Dict[str, Any]]:
    f"""
    Generate {num_conversations} conversations:
      - 9 covering every (clarity, req_orient) pair in scores^2
      - {num_conversations - 9} with random pairs
    
    Returns:
        Tuple of (list of written file paths, timing statistics dictionary)
    """
    set_seed(seed)
    
    # Start total timer
    total_start_time = time.time()

    feature_handler = FeatureHandler.from_json(features_config)
    conv_generator = CCConversationGenerator(feature_handler=feature_handler)

    context = {
        "location": {
            "position": [48.2628, 11.6687],
            "address": "Am Parkring, Munich, Germany",
            "data": "2025-03-19T0",
            "time": "09:00:00",
        },
        "person": {"gender": "male", "age": 51},
    }

    # Build the {num_conversations} dimension pairs
    all_pairs = list(product(scores, repeat=2))  # 9 exhaustive
    random_pairs = [
        (random.randint(0, 2), random.randint(0, 2)) for _ in range(num_conversations - len(all_pairs))
    ]
    dimension_pairs = all_pairs + random_pairs
    dimension_pairs = dimension_pairs[:num_conversations]  # In case num_conversations < 9

    print(dimension_pairs)

    # Output directory
    out_dir = Path(output_dir) if output_dir else Path(__file__).parent / "new_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    timing_data: List[Dict[str, Any]] = []

    for idx, (clarity, req_orient) in enumerate(dimension_pairs, start=1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(dimension_pairs)}] Generating conversation: "
              f"Clarity={clarity}, Request-orientedness={req_orient}")
        print(f"{'='*60}")

        # Start timer for this conversation
        conv_start_time = time.time()

        # Set dimension targets on the IPA class
        IPADims._clarity_level = clarity
        IPADims._request_orientedness_level = req_orient

        # Sample feature variables & generate a conversation skeleton
        sampled = feature_handler.sample_feature_scores()
        ordinal_vars = list(sampled.ordinal)
        categorical_vars = list(sampled.categorical)
        continuous_vars = list(sampled.continuous) if sampled.continuous else []

        conversation = conv_generator.generate_conversation(
            ordinal_vars=ordinal_vars,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
        )

        # Wrap for simulate_conversation interface: List[List[Conversation]]
        wrapped = [[conversation]]

        results = IPADims.simulate_conversation(
            list_individuals=wrapped,
            variable_names=["conversation"],
            scenario_path=os.getcwd(),
            sim_time=0,
            context=context,
            config_path=features_config,
            max_retries=3,
            min_turns=min_turns,
            max_turns=max_turns,
            max_repeats=max_repeats,
        )

        # End timer for this conversation
        conv_end_time = time.time()
        conv_duration = conv_end_time - conv_start_time

        if not results:
            print(f"No result for conversation {idx}, skipping.")
            timing_data.append({
                "conversation_id": idx,
                "clarity": clarity,
                "request_orientedness": req_orient,
                "generation_time_seconds": round(conv_duration, 2),
                "num_turns": 0,
                "average_turn_time_seconds": 0.0,
                "status": "failed",
            })
            continue

        sim_out = results[0]
        conv = sim_out.conversation

        n_turns = len(conv.turns)
        avg_turn_time = conv_duration / n_turns if n_turns > 0 else 0.0

        # Serialise
        conv_json = conversation_to_json(
            conversation=conv,
            individual_id=idx,
            clarity=clarity,
            request_orientedness=req_orient,
            generation_time_seconds=conv_duration,
            num_turns=n_turns,
        )

        # Try to fill in style_input from feature values
        try:
            fv = feature_handler.get_feature_values_dict(
                ordinal_feature_scores=ordinal_vars,
                categorical_feature_indices=categorical_vars,
            )
            style_desc = StyleDescription.model_validate(fv)
            conv_json["metadata"]["style_input"] = {
                k: v for k, v in style_desc.model_dump().items()
                if v is not None
            }
        except Exception:
            pass

        # Try to fill content_input more completely from the initial content
        try:
            fv = feature_handler.get_feature_values_dict(
                ordinal_feature_scores=ordinal_vars,
                categorical_feature_indices=categorical_vars,
            )
            ci = CCContentInput.model_validate(fv)
            conv_json["metadata"]["content_input"] = {
                k: v for k, v in ci.model_dump().items()
                if v is not None
            }
        except Exception:
            pass

        # Save
        conv_path = out_dir / f"conv_{idx:04d}.json"
        conv_path.write_text(
            json.dumps(conv_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(conv_path)
        
        # Store timing data for this conversation
        timing_data.append({
            "conversation_id": idx,
            "clarity": clarity,
            "request_orientedness": req_orient,
            "generation_time_seconds": round(conv_duration, 2),
            "num_turns": n_turns,
            "average_turn_time_seconds": round(avg_turn_time, 2),
            "status": "success",
            "file": f"conv_{idx:04d}.json",
        })
        
        print(f"  Saved {n_turns}-turn conversation to {conv_path}")
        print(f"  Total generation time: {conv_duration:.2f} seconds")
        print(f"  Average time per turn: {avg_turn_time:.2f} seconds")

    # End total timer
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Calculate statistics
    successful_times = [t["generation_time_seconds"] for t in timing_data if t["status"] == "success"]
    successful_convs = [t for t in timing_data if t["status"] == "success"]
    total_turns = sum(t["num_turns"] for t in successful_convs)
    total_turn_time = sum(t["generation_time_seconds"] for t in successful_convs)
    avg_time_per_turn_global = total_turn_time / total_turns if total_turns > 0 else 0
    
    timing_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_conversations": len(dimension_pairs),
        "successful_conversations": len(successful_times),
        "failed_conversations": len(dimension_pairs) - len(successful_times),
        "total_time_seconds": round(total_duration, 2),
        "total_time_formatted": f"{int(total_duration // 60)}m {int(total_duration % 60)}s",
        "average_time_per_conversation": round(sum(successful_times) / len(successful_times), 2) if successful_times else 0,
        "min_time_per_conversation": round(min(successful_times), 2) if successful_times else 0,
        "max_time_per_conversation": round(max(successful_times), 2) if successful_times else 0,
        "total_turns_generated": total_turns,
        "average_time_per_turn": round(avg_time_per_turn_global, 2),
        "conversations": timing_data,
    }
    
    # Save timing summary
    timing_path = out_dir / "generation_timing.json"
    timing_path.write_text(
        json.dumps(timing_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Generated {len(written)} conversations in {out_dir}")
    print(f"⏱  Total time: {timing_summary['total_time_formatted']}")
    print(f"⏱  Average time per conversation: {timing_summary['average_time_per_conversation']:.2f}s")
    print(f"⏱  Average time per turn (global): {timing_summary['average_time_per_turn']:.2f}s")
    print(f"⏱  Total turns generated: {timing_summary['total_turns_generated']}")
    print(f"⏱  Timing details saved to {timing_path}")
    print(f"{'='*60}")
    
    return written, timing_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate conversations with controlled dimension scores "
                    "using SYSTEM_PROMPT_CONTENT_INPUT_HISTORY_DIMS."
    )
    parser.add_argument(
        "--features_config",
        type=str,
        default="configs/features_simple_judge_cc.json",
        help="Path to features configuration JSON.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--min_turns", type=int, default=3,
        help="Minimum turns per conversation (default: 3).",
    )
    parser.add_argument(
        "--max_turns", type=int, default=6,
        help="Maximum turns per conversation (default: 6).",
    )
    parser.add_argument(
        "--max_repeats", type=int, default=2,
        help="Maximum repeat intents per conversation (default: 2).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: examples/car_control/carcontrol-survey/new_data/).",
    )
    parser.add_argument(
        "--llm_intent_classifier", type=str, default=None,
        help="Override LLM_IPA for intent classification.",
    )
    parser.add_argument(
        "--llm_generator", type=str, default=None,
        help="Override LLM_TYPE for utterance generation.",
    )
    parser.add_argument(
        "--num_conversations", type=int, default=15,
        help="Total number of conversations to generate (default: 15).",
    )   
    parser.add_argument(
        "--scores",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="List of scores to use for dimension pairs (default: [0, 1, 2]).",
    )
    args = parser.parse_args()

    # Override LLM configs if requested
    import llm.config as llm_config
    if args.llm_intent_classifier:
        llm_config.LLM_IPA = args.llm_intent_classifier
    if args.llm_generator:
        llm_config.LLM_TYPE = args.llm_generator

    paths, timing_summary = generate_all_conversations(
        features_config=args.features_config,
        seed=args.seed,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_repeats=args.max_repeats,
        output_dir=args.output_dir,
        num_conversations=args.num_conversations,
        scores=args.scores,
    )

    print("\nGenerated files:")
    for p in paths:
        print(f"  {p}")
    
    print(f"\n📊 Timing Summary:")
    print(f"  Total time: {timing_summary['total_time_formatted']}")
    print(f"  Successful: {timing_summary['successful_conversations']}/{timing_summary['total_conversations']}")
    print(f"  Average per conversation: {timing_summary['average_time_per_conversation']:.2f}s")
    print(f"  Average per turn: {timing_summary['average_time_per_turn']:.2f}s")
    print(f"  Range: {timing_summary['min_time_per_conversation']:.2f}s - {timing_summary['max_time_per_conversation']:.2f}s")
    print(f"  Total turns: {timing_summary['total_turns_generated']}")


if __name__ == "__main__":
    main()