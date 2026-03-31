from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from examples.navi.models import NaviContentInput, StyleDescription
from examples.navi.navi_utterance_generator import NaviUtteranceGenerator
from llm.features import FeatureHandler
from llm.features.models import FeatureType


JUDGE_DIMENSIONS: Dict[str, Any] = {
    "Clarity": {
        "description": "The system responses should be clear and concise.",
        "scores": {
            0: "System responses are unclear, verbose, or confusing throughout the whole conversation.",
            1: "System responses are partially unclear or verbose throughout the whole conversation, requiring effort to extract useful information.",
            2: "System responses are clear and concise throughout the whole conversation; user can easily understand them.",
        },
    },
    "Request-orientedness": {
        "description": "The system addresses the user's goal or responds appropriately, even if the answer is negative.",
        "scores": {
            0: "System doesn't address the request at all; response is completely unrelated to the user request.",
            1: "System partially addresses the request; somewhat related but no concrete POI/navigation offered, or asks for more info without having searched.",
            2: "System fully addresses the request; contains a relevant POI or clearly references the user's goal. A clear negative answer is acceptable if no suitable POI is available.",
        },
    },
}


class NaviFeatureSampler:
    """
    Samples 1..N Navi instances: content_input + style_input (constrained).
    Also samples judge scores in {0,1,2} for both Clarity and Request-orientedness.
    """

    def __init__(
        self,
        feature_handler: FeatureHandler,
        apply_constrains_to_vars: bool = True,
        rng: random.Random | None = None,
    ):
        self.feature_handler = feature_handler
        self.apply_constrains_to_vars = apply_constrains_to_vars
        self.rng = rng or random.Random()

        # Reuse constraint logic from utterance generator (no RAG, no LLM call)
        self._gen = NaviUtteranceGenerator(
            feature_handler=feature_handler,
            apply_constrains_to_vars=apply_constrains_to_vars,
            use_rag=False,
        )

    def _update_vars_from_model(
        self,
        ordinal_vars: List[float],
        categorical_vars: List[int],
        model_obj: object,
    ) -> tuple[List[float], List[int]]:
        # Mirror NaviUtteranceGenerator._update_features_from_content_input()
        for i, feature in enumerate(self.feature_handler.ordinal_features.values()):
            if not hasattr(model_obj, feature.name):
                continue
            new_value = getattr(model_obj, feature.name, None)
            new_var = self.feature_handler.get_var_from_feature_value(
                feature, new_value, feature_type=FeatureType.ORDINAL
            )
            if new_var is not None:
                ordinal_vars[i] = new_var

        for i, feature in enumerate(self.feature_handler.categorical_features.values()):
            if not hasattr(model_obj, feature.name):
                continue
            new_value = getattr(model_obj, feature.name, None)
            new_var = self.feature_handler.get_var_from_feature_value(
                feature, new_value, feature_type=FeatureType.CATEGORICAL
            )
            if new_var is not None:
                categorical_vars[i] = new_var

        return ordinal_vars, categorical_vars

    def _sample_judge_scores(self) -> Dict[str, int]:
        return {
            "Clarity": self.rng.choice([0, 1, 2]),
            "Request-orientedness": self.rng.choice([0, 1, 2]),
        }

    def _sample_num_turns(self) -> int:
        return self.rng.randint(2, 8)

    def sample_one(self) -> Dict[str, Any]:
        vars_ = self.feature_handler.sample_feature_scores()
        ordinal_vars: List[float] = list(vars_.ordinal)
        categorical_vars: List[int] = list(vars_.categorical)

        feature_values = self.feature_handler.get_feature_values_dict(
            ordinal_feature_scores=ordinal_vars,
            categorical_feature_indices=categorical_vars,
        )

        content_input = NaviContentInput.model_validate(feature_values)
        style_input = StyleDescription.model_validate(feature_values)

        # Apply constraints exactly like the generator
        content_input = self._gen.apply_constraints(content_input)
        style_input = self._gen.apply_constraints_style(content_input, style_input)

        # Keep vars consistent with constrained values (optional)
        if self.apply_constrains_to_vars:
            ordinal_vars, categorical_vars = self._update_vars_from_model(
                ordinal_vars, categorical_vars, content_input
            )
            ordinal_vars, categorical_vars = self._update_vars_from_model(
                ordinal_vars, categorical_vars, style_input
            )

        # Return with judge_dimensions first
        return {
            "judge_dimensions": self._sample_judge_scores(),
            "num_turns": self._sample_num_turns(),
            "content_input": content_input.model_dump(exclude_none=True),
            "style_input": style_input.model_dump(exclude_none=True),
        }

    def sample(self, n: int) -> List[Dict[str, Any]]:
        return [self.sample_one() for _ in range(n)]


def write_navi_samples_to_folder(
    n: int,
    features_config_path: str = "configs/navi_features.json",
    output_dir: str | Path | None = None,
    filename_prefix: str = "navi_sample_",
) -> List[Path]:
    """
    Create N JSON files on disk.

    Default: write to the `navi_sampler/` folder (same folder as this file).
    """
    fhandler = FeatureHandler.from_json(features_config_path)
    sampler = NaviFeatureSampler(fhandler, apply_constrains_to_vars=True)

    # Default: store to navi_sampler/ (this file's folder)
    out_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for i in range(1, n + 1):
        payload = sampler.sample_one()
        path = out_dir / f"{filename_prefix}{i:04d}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(path)

    return written


if __name__ == "__main__":
    # Produces specified number of JSON files in examples/navi/navi_sampler/
    paths = write_navi_samples_to_folder(
        n=30,
        features_config_path="configs/features_simple_judge_industry.json",
        output_dir=Path(__file__).parent,
    )
    print("\n".join(str(p) for p in paths))