import sys

from eval.navi.models import NaviContentInput, NaviContentOutput

sys.path.insert(0, "../opensbt-llm/")

from collections import defaultdict
from typing import Tuple, List, Dict, Optional

import numpy as np

from examples.navi.fitness import NaviFitnessContentComparison
from llm.config import N_VALIDATORS
from judge_eval.validator_dim import llm_validator_conversation
from opensbt.evaluation.fitness import Fitness
from llm.model.mt_simout import MultiTurnSimulationOutput
from llm.eval.fitness import counter_validations

from examples.navi.fitness_mt import NaviFitnessConversationEffectiveness, NaviFitnessConversationEfficiency, NaviFitnessConversationValidationDimensions
from llm.eval.critical import CriticalByFitnessThreshold, CriticalMerged
from llm.eval.fitness import FitnessMerged

from examples.car_control.fitness_mt import CCFitnessConversationValidationDimensions

def get_fitness_fnc(llm_type = "gpt-5-mini", weights=[0.5, 0.5], dimension_labels = ["C", "R"], max_score=2):
    return FitnessMerged([
        NaviFitnessConversationValidationDimensions(llm_type = llm_type, weights=weights, dimension_labels=dimension_labels, max_score=max_score),
        # NaviFitnessConversationEfficiency(),
        # NaviFitnessConversationEffectiveness(),
    ])

def get_fitness_fnc_carcontrol(llm_type = "gpt-5-mini", weights=[0.5, 0.5], dimension_labels = ["C", "R"], max_score=2):
    return FitnessMerged([
        CCFitnessConversationValidationDimensions(llm_type = llm_type, weights=weights, dimension_labels=dimension_labels, max_score=max_score),
        # CCFitnessConversationEfficiency(),
        # CCFitnessConversationEffectiveness(),
    ])

def get_critical_fnc(fitness_fnc, score_threshold=0.7):
    return CriticalMerged(
        fitness_names=fitness_fnc.name,
        criticals=[
            (CriticalByFitnessThreshold(mode = "<", score=score_threshold), ["dimensions_fitness"]),
            # (CriticalByFitnessThreshold(mode = "<", score=score_threshold), ["efficiency_fitness"]),
            # (CriticalByFitnessThreshold(mode = "<", score=score_threshold), ["effectiveness_fitness"]),
        ],
        mode="or",
    )
def get_critical_fnc_carcontrol(fitness_fnc, score_threshold=0.7):
    return CriticalMerged(
        fitness_names=fitness_fnc.name,
        criticals=[
            (CriticalByFitnessThreshold(mode = "<", score=score_threshold), ["dimensions_fitness"]),
            # (CriticalByFitnessThreshold(mode = "<", score=score_threshold), ["efficiency_fitness"]),
            # (CriticalByFitnessThreshold(mode = "<", score=score_threshold), ["effectiveness_fitness"]),
        ],
        mode="or",
    )