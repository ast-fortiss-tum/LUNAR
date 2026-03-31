import dataclasses
import pydantic

from opensbt.experiment.search_configuration import SearchConfiguration, SearchOperators
from llm.operators.base_classes import (
    ConversationRepairBase,
    UtteranceCrossoverBase,
    UtteranceMutationBase,
    UtteranceSamplingBase,
    UtteranceDuplicateEliminationBase,
    UtteranceRepairBase,
    ConversationCrossoverBase,
    ConversationMutationBase,
    ConversationSamplingBase,
    ConversationDuplicateEliminationBase,
)
from llm.operators.utterance_crossover import NoUtteranceCrossover
from llm.operators.utterance_mutator import UtteranceMutation
from llm.operators.utterance_sampling import UtteranceSampling
from llm.operators.utterance_duplicates import UtteranceDuplicateElimination
from llm.operators.utterance_repair import NoUtteranceRepair
from llm.operators.conversation_sampling_discrete import ConversationSamplingDiscrete
from llm.operators.conversation_mutator_discrete import ConversationMutationDiscrete
from llm.operators.conversation_crossover_discrete import ConversationCrossoverDiscrete
from llm.operators.conversation_duplicates import ConversationDuplicateEliminationLocal
from llm.operators.conversation_repair import ConversationRepairConversationGenerator


class QASearchOperators(SearchOperators):
    crossover: UtteranceCrossoverBase = NoUtteranceCrossover()
    sampling: UtteranceSamplingBase = UtteranceSampling()
    duplicate_elimination: UtteranceDuplicateEliminationBase = UtteranceDuplicateElimination()
    mutation: UtteranceMutationBase = UtteranceMutation()
    repair: UtteranceRepairBase = NoUtteranceRepair()


class QASearchConfiguration(SearchConfiguration):
    operators: QASearchOperators = QASearchOperators()


class MultiTurnSearchOperators(SearchOperators):
    crossover: ConversationCrossoverBase = ConversationCrossoverDiscrete()
    sampling: ConversationSamplingBase = ConversationSamplingDiscrete()
    duplicate_elimination: ConversationDuplicateEliminationBase = ConversationDuplicateEliminationLocal()
    mutation: ConversationMutationBase = ConversationMutationDiscrete()
    repair: ConversationRepairBase = ConversationRepairConversationGenerator()


class MultiTurnSearchConfiguration(SearchConfiguration):
    operators: MultiTurnSearchOperators = MultiTurnSearchOperators()