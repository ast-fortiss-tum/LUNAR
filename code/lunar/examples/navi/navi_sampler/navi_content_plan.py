import random
from typing import Dict, Any, Optional, Set

from examples.navi.models import NaviContentInput
from examples.navi.navi_utterance_generator import NaviUtteranceGenerator


def sample_constrained_conversation_plan(feature_handler) -> Dict[str, Any]:
    """
    Samples categorical vars -> decodes to values -> applies Navi constraints -> returns dict.
    """
    sampled_categorical_vars = feature_handler._sample_features(feature_handler.categorical_features.values())
    
    plan_feature_values = feature_handler.get_feature_values_dict(
        ordinal_feature_scores=[],
        categorical_feature_indices=sampled_categorical_vars,
    )
    plan_categorical = {k: plan_feature_values.get(k, None) for k in feature_handler.categorical_features.keys()}

    ci = NaviContentInput.model_validate(plan_categorical)
    ci = NaviUtteranceGenerator(feature_handler=feature_handler).apply_constraints(ci)

    constrained_plan = {k: getattr(ci, k, None) for k in feature_handler.categorical_features.keys()}
    
    return constrained_plan


def build_content_requirements_text(
    content_input: Optional[NaviContentInput],
    used_features: Optional[Set[str]] = None,
    new_feature_this_turn: Optional[str] = None,
) -> str:
    """
    Human-readable requirements to be inserted into prompts.
    
    If used_features is provided, only show category + used features + new feature.
    """
    if content_input is None:
        return "- No additional mandatory content constraints."

    d = content_input.model_dump(exclude_none=True)
    if not d:
        return "- No additional mandatory content constraints."

    if used_features is not None:
        # Filter to only show relevant features
        filtered = {}
        if "category" in d:
            filtered["category"] = d["category"]
        for k in used_features:
            if k in d:
                filtered[k] = d[k]
        if new_feature_this_turn and new_feature_this_turn in d:
            filtered[new_feature_this_turn] = d[new_feature_this_turn]
        d = filtered

    if not d:
        return "- No additional mandatory content constraints."

    lines = []
    for k, v in d.items():
        lines.append(f"- {k}: {v}")
    lines.append("- You MUST reflect every listed attribute in your turn (explicitly or implicitly).")
    return "\n".join(lines)


def constrained_turn_content_input(
    feature_handler,
    conversation_plan: Dict[str, Any],
    features_to_include: Set[str],
) -> NaviContentInput:
    """
    Build a NaviContentInput for a single turn from the conversation_plan.

    Rules:
    - Always include category (even if not in features_to_include).
    - Include only features listed in features_to_include (plus category).
    """
    values: Dict[str, Any] = {}

    # always include category
    values["category"] = conversation_plan.get("category", None)

    # include selected features (e.g., rating) from the plan
    for feat in features_to_include:
        # keep only keys that exist in the plan
        if feat in conversation_plan:
            values[feat] = conversation_plan.get(feat, None)


    return NaviContentInput.model_validate(values)


def pick_random_subset_keep_one_unused(eligible: list[str]) -> Set[str]:
    """
    Returns a random subset of eligible features, with constraint:
    if len(eligible) >= 2: choose k in [0, len(eligible)-1] (leave at least one unused).
    else: k == len(eligible)
    """
    n = len(eligible)
    if n >= 2:
        k = random.randint(0, n - 1)  # leave at least one unused
    else:
        k = n
    if k == 0:
        return set()
    
    return set(random.sample(eligible, k=k))


def pick_one_unused_feature(
    conversation_plan: Dict[str, Any],
    used: Set[str],
    current_content_input: Optional[Any] = None,
    feature_handler: Optional[Any] = None,
) -> Optional[str]:
    """
    Picks exactly one unused (and non-None) feature from the plan (excluding category).
    
    Price constraint: fuel_price is excluded if it is already at the lowest
    possible value (cannot be lowered further).
    """
    candidates = [
        k for k, v in conversation_plan.items()
        if k != "category" and v is not None and k not in used
    ]

    # price constraint
    if (
        "fuel_price" in candidates
        and current_content_input is not None
        and feature_handler is not None
    ):
        current_price = getattr(current_content_input, "fuel_price", None)
        if current_price is not None:
            feat = (
                feature_handler.categorical_features.get("fuel_price")
                or feature_handler.ordinal_features.get("fuel_price")
            )
            if feat is not None:
                lower_options = [v for v in feat.values if v < current_price]
                if not lower_options:
                    candidates.remove("fuel_price")

    if not candidates:
        return None
    
    return random.choice(candidates)