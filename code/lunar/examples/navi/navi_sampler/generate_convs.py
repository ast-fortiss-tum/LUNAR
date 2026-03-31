"""
Generate multi-turn conversations using:
1. NaviFeatureSampler to create feature configurations (content + style + judge dimensions)
2. LLM to generate conversations based on the GENERATION_PROMPT template

Usage:
    python generate_convs.py --n 30 --output_dir ./convs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .navi_sampler import NaviFeatureSampler, JUDGE_DIMENSIONS
from .llm_gen_prompt import GENERATION_PROMPT
from llm.features import FeatureHandler
from llm.llms import pass_llm, LLMType
from json_repair import repair_json


# Load dimension definitions and dimension examples from files
DIMENSIONS_DEFINITIONS_FILE = Path(__file__).parent / "dimensions_definitions.txt"
DIMENSIONS_EXAMPLES_FILE = Path(__file__).parent / "dimensions_examples.txt"


def load_file_content(filepath: Path) -> str:
    """Load content from a text file."""
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


def format_judge_dimensions_for_prompt() -> str:
    """
    Format the JUDGE_DIMENSIONS dictionary for inclusion in the prompt.
    Includes descriptions and score meanings to guide the LLM.
    """
    lines = []
    for dim_name, dim_info in JUDGE_DIMENSIONS.items():
        lines.append(f"**{dim_name}**")
        lines.append(f"Description: {dim_info['description']}")
        lines.append("Score meanings:")
        for score, meaning in dim_info['scores'].items():
            lines.append(f"  - {score}: {meaning}")
        lines.append("")
    
    return "\n".join(lines)


def build_generation_prompt(sample_json: Dict[str, Any]) -> str:
    """
    Build the full prompt for conversation generation by combining:
    - The generation prompt template
    - The dimension definitions
    - The dimension examples
    - The sampled JSON configuration
    """
    dimensions_definitions = load_file_content(DIMENSIONS_DEFINITIONS_FILE)
    dimensions_examples = load_file_content(DIMENSIONS_EXAMPLES_FILE)
    
    # Build the input JSON - keep it simple with just the scores
    input_json_str = json.dumps(sample_json, indent=2, ensure_ascii=False)
    
    full_prompt = GENERATION_PROMPT.format(
        dimensions_definitions=dimensions_definitions,
        dimensions_examples=dimensions_examples,
        input_json=input_json_str
    )
    
    return full_prompt


def extract_tagged_content(text: str, tag: str) -> str:
    """
    Extract content between XML-like tags.
    
    Args:
        text: The full text containing tagged content
        tag: The tag name (e.g., 'generation_plan' or 'conversation')
    
    Returns:
        The content between the opening and closing tags, or empty string if not found
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def parse_conversation_to_structured(conversation_text: str) -> List[Dict[str, str]]:
    """
    Parse plain text conversation in "User:/System:" format into structured turns.
    
    Args:
        conversation_text: Plain text conversation like:
            User: hello
            System: hi there
            User: find coffee
            System: here are some options
    
    Returns:
        List of turn dictionaries with 'user' and 'system' keys:
        [
            {"user": "hello", "system": "hi there"},
            {"user": "find coffee", "system": "here are some options"}
        ]
    """
    turns = []
    text = conversation_text.strip()
    
    # Split by "User:" - this will include the first User: marker
    user_sections = re.split(r'(?:^|\n)User:\s*', text)
    
    # Remove empty first element if conversation starts with "User:"
    if not user_sections[0].strip():
        user_sections = user_sections[1:]
    
    for user_section in user_sections:
        if not user_section.strip():
            continue
        
        # Split by "System:" to separate user utterance from system response
        parts = re.split(r'\nSystem:\s*', user_section.strip(), maxsplit=1)
        
        user_utterance = parts[0].strip()
        
        if len(parts) == 2:
            # Has both user and system
            system_utterance = parts[1].strip()
        else:
            # Only user utterance (edge case)
            system_utterance = ""
        
        turns.append({
            "user": user_utterance,
            "system": system_utterance
        })
    
    return turns


def generate_conversation(
    sample_json: Dict[str, Any],
    llm_type: LLMType = LLMType.GPT_4O,
    max_retries: int = 3,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Generate a conversation from a sampled feature configuration using an LLM.
    
    Args:
        sample_json: The sampled feature configuration from NaviFeatureSampler
        llm_type: The LLM to use for generation
        max_retries: Number of retries on failure
        temperature: Temperature for LLM generation (0.0-1.0)
    
    Returns:
        The generated conversation as a dictionary with structured turns and generation plan
    """
    prompt = build_generation_prompt(sample_json)
    
    for attempt in range(max_retries):
        try:
            response = pass_llm(
                msg=prompt,
                max_tokens=2000,
                temperature=temperature,
                llm_type=llm_type,
                system_message="Generate a multi-turn conversation based ONLY on the provided instruction."
            )
            
            # Extract generation plan and conversation from tagged sections
            generation_plan = extract_tagged_content(response, "generation_plan")
            conversation_text = extract_tagged_content(response, "conversation")
            
            # Parse the conversation into structured format
            turns = parse_conversation_to_structured(conversation_text)
            
            conversation_data = {
                "generation_plan": generation_plan,
                "turns": turns,
                "raw_conversation": conversation_text,
                "raw_response": response.strip(),
                "metadata": sample_json
            }
            
            return conversation_data
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                # Return a structured error response
                return {
                    "error": str(e),
                    "raw_response": response if 'response' in locals() else None,
                    "metadata": sample_json
                }
    
    return {"error": "Max retries exceeded", "metadata": sample_json}


def generate_conversations(
    n: int,
    features_config_path: str = "configs/features_simple_judge_industry.json",
    output_dir: str | Path | None = None,
    llm_type: LLMType = LLMType.GPT_4O,
    save_samples: bool = True,
    temperature: float = 0.7,
) -> List[Path]:
    """
    Generate N conversations and save them to disk.
    
    Args:
        n: Number of conversations to generate
        features_config_path: Path to the features configuration JSON
        output_dir: Directory to save conversations (default: same folder as this script)
        llm_type: LLM to use for conversation generation
        save_samples: Whether to also save the intermediate sample JSONs
        temperature: Temperature for LLM generation (0.0-1.0)
    
    Returns:
        List of paths to the saved conversation files
    """
    # Initialize feature handler and sampler
    fhandler = FeatureHandler.from_json(features_config_path)
    sampler = NaviFeatureSampler(fhandler, apply_constrains_to_vars=True)
    
    # Setup output directory
    out_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent / "convs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create samples directory if saving samples
    if save_samples:
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
    
    written_paths: List[Path] = []
    
    print(f"Generating {n} conversations using {llm_type.value} (temperature={temperature})...")
    
    for i in range(1, n + 1):
        print(f"\n[{i}/{n}] Sampling features...")
        
        # Step 1: Sample feature configuration
        sample_json = sampler.sample_one()
        
        # Save sample if requested
        if save_samples:
            sample_path = samples_dir / f"sample_{i:04d}.json"
            sample_path.write_text(
                json.dumps(sample_json, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"  Saved sample to: {sample_path}")
        
        # Step 2: Generate conversation using LLM
        print(f"  Generating conversation with LLM...")
        conversation = generate_conversation(sample_json, llm_type=llm_type, temperature=temperature)
        
        # Step 3: Save conversation
        conv_path = out_dir / f"conv_{i:04d}.json"
        conv_path.write_text(
            json.dumps(conversation, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        written_paths.append(conv_path)
        
        if "error" in conversation:
            print(f"  ⚠ Saved with error: {conversation['error']}")
        else:
            # Count turns from structured data
            num_turns = len(conversation.get("turns", []))
            has_plan = bool(conversation.get("generation_plan"))
            print(f"  ✓ Saved conversation with {num_turns} turns{' and generation plan' if has_plan else ''} to: {conv_path}")
    
    print(f"\n✓ Generated {len(written_paths)} conversations in {out_dir}")
    return written_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn conversations using feature sampling and LLM generation."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of conversations to generate (default: 30)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated conversations (default: ./convs)"
    )
    parser.add_argument(
        "--features_config",
        type=str,
        default="configs/features_simple_judge_industry.json",
        help="Path to the features configuration JSON"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="GPT_4O",
        choices=[e.name for e in LLMType],
        help="LLM type to use for generation (default: GPT_4O)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM generation, 0.0-1.0 (default: 0.7)"
    )
    parser.add_argument(
        "--no_save_samples",
        action="store_true",
        help="Do not save intermediate sample JSONs"
    )
    
    args = parser.parse_args()
    
    # Convert LLM string to enum
    llm_type = LLMType[args.llm]
    
    # Generate conversations
    paths = generate_conversations(
        n=args.n,
        features_config_path=args.features_config,
        output_dir=args.output_dir,
        llm_type=llm_type,
        save_samples=not args.no_save_samples,
        temperature=args.temperature,
    )
    
    print("\nGenerated files:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()