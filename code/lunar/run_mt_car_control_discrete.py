import os
os.environ["WANDB__SERVICE"] = "false"

import argparse
from datetime import datetime

import pymoo
import wandb

import weave

from examples.car_control.fitness_mt import CCFitnessConversationEffectiveness, CCFitnessConversationEfficiency, CCFitnessConversationValidationDimensions
from llm.eval.critical import CriticalByFitnessThreshold, CriticalMerged
from llm.eval.fitness import FitnessMerged
from llm.llm_output import ExtendedEncoder
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.nsga2d_optimizer import NSGAIIDOptimizer
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.ps_rand import PureSamplingRand

from opensbt.utils.log_utils import log, setup_logging, disable_pymoo_warnings
from opensbt.config import RESULTS_FOLDER, LOG_FILE
from opensbt.utils.wandb import logging_callback_archive

from llm.model.qa_problem import QAProblem
from llm.adapter.embeddings_openai_adapter import get_similarity_conversation

# Discrete Operators
from llm.operators.conversation_sampling_discrete import ConversationSamplingDiscrete
from llm.operators.conversation_mutator_discrete import ConversationMutationDiscrete
from llm.operators.conversation_crossover_discrete import ConversationCrossoverDiscrete
from llm.operators.conversation_duplicates import ConversationDuplicateEliminationLocal, ConversationDuplicateEliminationVars
from llm.operators.conversation_repair import ConversationRepairConversationGenerator, NoConversationRepair
from examples.car_control.cc_conversation_generator import CCConversationGenerator
from llm.features import FeatureHandler
from llm.model.search_configuration import MultiTurnSearchConfiguration, MultiTurnSearchOperators
from llm.sut.ipa import IPA
from llm.sut.ipa_industry_cc import IndustryIPA
from llm.sut.ipa_yelp_cc import IPA_YELP
from llm.sut.ipa_los import IPA_LOS
from llm.llms import LLMType
from mt_navi_runs_utils import save_results_to_json

import logging

# Setup Environment
weave_cache_dir = os.environ.get("WEAVE_CACHE_DIR")
os.environ["WEAVE_CACHE_DIR"] = os.getcwd() + os.sep + "/wandb-cache/"

# CLI
parser = argparse.ArgumentParser(description="Run multi-turn Conversation optimization pipeline.")
parser.add_argument("--algorithm", type=str, choices=["rs", "nsga2", "nsga2d"], default="nsga2d", help="Algorithm.")
parser.add_argument("--n", type=int, default=6, help="Population size.")
parser.add_argument("--i", type=int, default=6, help="Number of generations.")
parser.add_argument("--archive_threshold", type=float, default=0.15, help="Threshold for novelty archive (NSGA-II-D).")
parser.add_argument("--sut", type=str, choices=["industry", "openai", "openai_los", "ipa_yelp"], default="ipa_yelp", help="System under test.")
parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")
parser.add_argument("--wandb_project", type=str, default="MultiTurnTestDiscrete", help="Weights & Biases project name.")
parser.add_argument(
    "--features_config",
    type=str,
    default="configs/features_simple_judge_cc.json",
    help="Path to features configuration JSON for NaviUtteranceGenerator / FeatureHandler.",
)
parser.add_argument("--weight_clarity", type=float, default=0.5, help="Weight for clarity dimension.")
parser.add_argument("--weight_request_orientedness", type=float, default=0.5, help="Weight for request-orientedness dimension.")
parser.add_argument("--th_dims", type=float, default=0.7, help="Critical threshold for dimensions fitness.")
parser.add_argument("--th_efficiency", type=float, default=0.7, help="Critical threshold for efficiency fitness.")
parser.add_argument("--th_effectiveness", type=float, default=0.7, help="Critical threshold for effectiveness fitness.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--use_repair",
    action="store_true",
    help="Use the repair operator instead of generating conversations during crossover and mutation",
)
parser.add_argument(
    "--max_time",
    type=str,
    default="00:01:30",
    help="Maximal execution time as string 'hh:mm:ss'. Use None to disable.",
)
parser.add_argument("--min_turns", type=int, default=3, help="Minimum number of turns in a conversation.")
parser.add_argument("--max_turns", type=int, default=6, help="Maximum number of turns in a conversation.")
parser.add_argument("--max_repeats", type=int, default=2, help="Maximum number of repeat intents allowed per conversation.")
parser.add_argument(
    "--store_turns_details",
    action="store_true",
    help="Store detailed turn-level results in a JSON file.",
)
parser.add_argument(
    "--llm_ipa",
    type=str,
    #default="DeepSeek-V3-0324",
    default="gpt-5-chat",
    help="LLM model for SUT",
)
parser.add_argument(
    "--llm_intent_classifier",
    type=str,
    #default="DeepSeek-V3-0324",
    default="gpt-4o",
    help="LLM model for system intent classification during conversation (overrides LLM_IPA from config).",
)
parser.add_argument(
    "--llm_judge",
    type=str,
    default="gpt-5-mini",
    help="LLM model for conversation dimension scoring (overrides LLM_VALIDATOR from config).",
)
parser.add_argument(
    "--llm_generator",
    type=str,
    default="gpt-5-mini",
    help="LLM model for utterance generation (overrides LLM_TYPE from config).",
)
parser.add_argument(
    "--save_folder",
    type=str,
    default=RESULTS_FOLDER,
    help="Path where to store results"
)

args = parser.parse_args()

import llm.config as llm_config
if args.llm_ipa:
    llm_config.LLM_IPA = args.llm_ipa
    os.environ["LLM_IPA"] = args.llm_ipa
    print(f"Set LLM_IPA to {args.llm_ipa}")
if args.llm_intent_classifier:
    llm_config.LLM_CLASSIFIER = args.llm_intent_classifier
    os.environ["LLM_CLASSIFIER"] = args.llm_intent_classifier
    print(f"Set LLM_CLASSIFIER to {args.llm_intent_classifier}")
if args.llm_judge:
    llm_config.LLM_VALIDATOR = args.llm_judge
    os.environ["LLM_VALIDATOR"] = args.llm_judge
    print(f"Set LLM_VALIDATOR to {args.llm_judge}")
if args.llm_generator:
    llm_config.LLM_GENERATOR = args.llm_generator
    os.environ["LLM_GENERATOR"] = args.llm_generator
    print(f"Set LLM_GENERATOR to {args.llm_generator}")

# Logging/setup
os.chmod(os.getcwd(), 0o777)
logger = log.getLogger(__name__)
setup_logging(LOG_FILE)
disable_pymoo_warnings()

# Suppress all Azure SDK logging
logging.getLogger("azure").setLevel(logging.WARNING)

# Optional: if you want to completely disable logs
logging.getLogger("azure").disabled = True

# load FeatureHandler once from CLI and reuse everywhere
feature_handler = FeatureHandler.from_json(args.features_config)

def create_problem_name(args, config) -> str:
    # update the name based on the sut used
    problem_name = (
        f"{args.algorithm.upper()}"
        + f"_{args.sut}"
        + f"_{args.llm_ipa}"
        + f"_{config.population_size}n"
        + (f"_{config.n_generations}i" if config.n_generations is not None else "")
        + (
            f"_{config.maximal_execution_time.replace(':', '_')}t"
            if config.maximal_execution_time is not None
            else ""
        )
        + f"_{args.seed}seed"
        # + f"_gen-{args.llm_generator}"
        # + f"_judge-{args.llm_judge}"
    )
    return problem_name

# Configure Discrete Operators
operators = MultiTurnSearchOperators(
    crossover=ConversationCrossoverDiscrete(
        generate_conversation=not args.use_repair
    ),
    sampling=ConversationSamplingDiscrete(
        generate_conversation=not args.use_repair,
        variable_length=True # assuming variable length
    ),
    mutation=ConversationMutationDiscrete(
        generate_conversation=not args.use_repair
    ),
    duplicate_elimination=ConversationDuplicateEliminationVars(), # ConversationDuplicateEliminationLocal(),
    repair=(
        ConversationRepairConversationGenerator()
        if args.use_repair
        else NoConversationRepair()
    ),
)


config = MultiTurnSearchConfiguration(operators=operators)
config.population_size = args.n
config.n_generations = args.i
config.archive_threshold = args.archive_threshold
config.maximal_execution_time = None if (args.max_time is None or str(args.max_time).lower() in ("none", "")) else args.max_time
config.n_repopulate_max = 0.2
config.results_folder = args.save_folder

#  SUT 
if args.sut == "industry":
    simulate_function = IndustryIPA.simulate_conversation
elif args.sut == "openai":
    simulate_function = IPA.simulate_conversation
elif args.sut == "openai_los":
    simulate_function = IPA_LOS.simulate_conversation
elif args.sut == "ipa_yelp":
    simulate_function = IPA_YELP.simulate_conversation
else:
    raise ValueError(f"Invalid SUT: {args.sut}. Valid options: industry, openai, openai_los, ipa_yelp")


fitness = FitnessMerged([
    CCFitnessConversationValidationDimensions(weights=[args.weight_clarity, args.weight_request_orientedness],),
    CCFitnessConversationEfficiency(),
    CCFitnessConversationEffectiveness(),
])

critical = CriticalMerged(
    fitness_names=fitness.name,
    criticals=[
        (CriticalByFitnessThreshold(mode = "<", score=args.th_dims), ["dimensions_fitness"]),
        (CriticalByFitnessThreshold(mode = "<", score=args.th_efficiency), ["efficiency_fitness"]),
        (CriticalByFitnessThreshold(mode = "<", score=args.th_effectiveness), ["effectiveness_fitness"]),
    ],
    mode="or",
)


problem = QAProblem(
    problem_name="MT_Test",
    scenario_path=os.getcwd(),
    xl=[0],
    xu=[1],
    simulation_variables=["conversation"],
    fitness_function=fitness,
    critical_function=critical,
    simulate_function=simulate_function,
    context={
        "location": {
            "position": [46.2628, 11.6687],  # NOTE: this location is for industry & openai_los - change it for conv_navi_yelp (Philadelphia)
            "address": "Munich, Germany",
            "data": "2025-03-19T0",
            "time": "09:00:00",
        },
        "person": {"gender": "male", "age": 51},
    },
    # # NOTE: this is for ipa_yelp
    # context={
    #     "location": {
    #         "position": [39.9555, -75.1999],
    #         "address": "38-98 S 39th St, Philadelphia, PA, USA",
    #         "data": "2025-03-19T0",
    #         "time": "09:00:00",
    #     },
    #     "person": {"gender": "female", "age": 33},
    # },
    seed=args.seed,
    min_turns=args.min_turns,
    max_turns=args.max_turns,
    max_repeats=args.max_repeats,
)

problem.feature_handler = feature_handler
problem.conversation_generator = CCConversationGenerator(feature_handler=feature_handler)

problem.problem_name = create_problem_name(args, config)

tags = [f"{k}:{v}" for k, v in vars(args).items() if k not in ("features_config", "save_folder")]

if not args.no_wandb:
    weave.init(args.wandb_project)
    wandb.init(
        entity="mt-test",
        project=args.wandb_project,
        name=problem.problem_name,
        group=datetime.now().strftime("%d-%m-%Y"),
        tags=tags
    )
else:
    wandb.init(mode="disabled")


if args.algorithm == "nsga2":
    optimizer = NsgaIIOptimizer(problem=problem, config=config, callback=logging_callback_archive)
elif args.algorithm == "nsga2d":
    optimizer = NSGAIIDOptimizer(
        problem=problem,
        config=config,
        callback=logging_callback_archive,
        dist_function=get_similarity_conversation,
    )
elif args.algorithm == "rs":
    optimizer = PureSamplingRand(problem=problem, config=config, callback=logging_callback_archive)
else:
    raise ValueError("Algorithm not known.")


res = optimizer.run()

if args.store_turns_details:
    save_results_to_json(res, args, problem, output_dir=optimizer.save_folder)
res.write_results(
    results_folder=optimizer.save_folder,
    params=optimizer.parameters,
    search_config=config
)
log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")
