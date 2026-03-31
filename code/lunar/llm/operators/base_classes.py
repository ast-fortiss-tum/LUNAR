import copy
import dataclasses
import typing as tp
from typing import List, Union, final, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pydantic
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling

from opensbt.operators import (
    CustomObjectCrossoverBase,
    CustomObjectDuplicateEliminationBase,
    CustomObjectMutationBase,
    CustomObjectOperatorBase,
    CustomObjectSamplingBase,
    CustomObjectRepairBase
)

from llm.model.models import Utterance, Conversation
from llm.model.qa_problem import QAProblem


@dataclasses.dataclass
class UtteranceOperatorBase(CustomObjectOperatorBase, ABC):
    @staticmethod
    def _has_style_features(utterance: Utterance) -> bool:
        return len(utterance.ordinal_vars) > 0
    
    @staticmethod
    def _has_content_features(utterance: Utterance) -> bool:
        return len(utterance.categorical_vars) > 0

    @staticmethod
    def _validate_instance(obj):
        assert isinstance(obj, Utterance), "Population must be made of Utterance instances"

    def _build_utterance(
        self,
        problem: QAProblem,
        seed: Optional[str],
        ordinal_vars: List[float],
        categorical_vars: List[int],
        ):
        if self.generate_question:
            utterance = problem.question_generator.generate_utterance(
                seed=seed,
                ordinal_vars=ordinal_vars,
                categorical_vars=categorical_vars,
                llm_type=self.llm_type,
            )
        else:
            utterance = Utterance(
                seed=seed,
                ordinal_vars=ordinal_vars,
                categorical_vars=categorical_vars,
            )
        return utterance


class UtteranceCrossoverBase(CustomObjectCrossoverBase, UtteranceOperatorBase, ABC):
    def _ordinal_vars_crossover(
        self,
        problem: Problem,
        matings: List[List[Utterance]],
        crossover: Crossover,
    ) -> List[List[List[float]]]:
        if self._has_style_features(matings[0][0]):
            return self._vars_crossover(
                problem, matings, crossover, attribute_name="ordinal_vars")
        else:
            return self._empty_crossover(len(matings))

    def _categorical_vars_crossover(
        self,
        problem: Problem,
        matings: List[List[Utterance]],
        crossover: Crossover,
    ) -> List[List[List[int]]]:
        if self._has_content_features(matings[0][0]):
            return self._vars_crossover(
                problem, matings, crossover, attribute_name="categorical_vars")
        else:
            return self._empty_crossover(len(matings))


class UtteranceMutationBase(CustomObjectMutationBase, UtteranceOperatorBase, ABC):
    def _ordinal_vars_mutation(
        self,
        problem: Problem,
        utterances: List[Utterance],
        mutation: Mutation,
    ) -> List[List[float]]:
        if self._has_style_features(utterances[0]):
            return self._vars_mutation(problem, utterances, mutation, "ordinal_vars")
        else:
            return self._empty_mutation(len(utterances))

    def _categoricals_vars_mutation(
        self,
        problem: Problem,
        utterances: List[Utterance],
        mutation: Mutation,
    ) -> List[List[int]]:
        if self._has_content_features(utterances[0]):
            return self._vars_mutation(problem, utterances, mutation, "categorical_vars")
        else:
            return self._empty_mutation(len(utterances))


class UtteranceSamplingBase(CustomObjectSamplingBase, UtteranceOperatorBase, ABC):
    pass


class UtteranceDuplicateEliminationBase(
    CustomObjectDuplicateEliminationBase, UtteranceOperatorBase, ABC
):
    pass


class UtteranceRepairBase(
    CustomObjectRepairBase, UtteranceOperatorBase, ABC
):
    pass


@dataclasses.dataclass
class ConversationOperatorBase(CustomObjectOperatorBase, ABC):
    @staticmethod
    def _has_style_features(conversation: Conversation) -> bool:
        return len(conversation.ordinal_vars) > 0

    @staticmethod
    def _has_content_features(conversation: Conversation) -> bool:
        return len(conversation.categorical_vars) > 0

    @staticmethod
    def _has_continuous_features(conversation: Conversation) -> bool:
        return len(conversation.continuous_vars) > 0

    @staticmethod
    def _validate_instance(obj):
        assert isinstance(obj, Conversation), "Population must be made of Conversation instances"

    def _build_conversation(
        self,
        problem: QAProblem,
        ordinal_vars: List[float],
        categorical_vars: List[int],
        continuous_vars: List[float],
        ):
        if self.generate_conversation:
            conversation = problem.conversation_generator.generate_conversation(
                ordinal_vars=ordinal_vars,
                categorical_vars=categorical_vars,
                continuous_vars=continuous_vars,
            )
        else:
            conversation = Conversation(
                ordinal_vars=ordinal_vars,
                categorical_vars=categorical_vars,
                continuous_vars=continuous_vars,
            )
        return conversation


class ConversationCrossoverBase(CustomObjectCrossoverBase, ConversationOperatorBase, ABC):
    def _ordinal_vars_crossover(
        self,
        problem: Problem,
        matings: List[List[Conversation]],
        crossover: Crossover,
    ) -> List[List[List[float]]]:
        if self._has_style_features(matings[0][0]):
            return self._vars_crossover(problem, matings, crossover, attribute_name="ordinal_vars")
        return self._empty_crossover(len(matings))

    def _categorical_vars_crossover(
        self,
        problem: Problem,
        matings: List[List[Conversation]],
        crossover: Crossover,
    ) -> List[List[List[int]]]:
        if self._has_content_features(matings[0][0]):
            return self._vars_crossover(problem, matings, crossover, attribute_name="categorical_vars")
        return self._empty_crossover(len(matings))

    def _continuous_vars_crossover(
        self,
        problem: Problem,
        matings: List[List[Conversation]],
        crossover: Crossover,
    ) -> List[List[List[float]]]:
        if self._has_continuous_features(matings[0][0]):
            return self._vars_crossover(problem, matings, crossover, attribute_name="continuous_vars")
        return self._empty_crossover(len(matings))


class ConversationMutationBase(CustomObjectMutationBase, ConversationOperatorBase, ABC):
    def _ordinal_vars_mutation(
        self,
        problem: Problem,
        conversations: List[Conversation],
        mutation: Mutation,
    ) -> List[List[float]]:
        if self._has_style_features(conversations[0]):
            return self._vars_mutation(problem, conversations, mutation, "ordinal_vars")
        return self._empty_mutation(len(conversations))

    def _categorical_vars_mutation(
        self,
        problem: Problem,
        conversations: List[Conversation],
        mutation: Mutation,
    ) -> List[List[int]]:
        if self._has_content_features(conversations[0]):
            return self._vars_mutation(problem, conversations, mutation, "categorical_vars")
        return self._empty_mutation(len(conversations))

    def _continuous_vars_mutation(
        self,
        problem: Problem,
        conversations: List[Conversation],
        mutation: Mutation,
    ) -> List[List[float]]:
        if self._has_continuous_features(conversations[0]):
            return self._vars_mutation(problem, conversations, mutation, "continuous_vars")
        return self._empty_mutation(len(conversations))


class ConversationSamplingBase(CustomObjectSamplingBase, ConversationOperatorBase, ABC):
    pass


class ConversationDuplicateEliminationBase(
    CustomObjectDuplicateEliminationBase, ConversationOperatorBase, ABC
):
    pass


class ConversationRepairBase(
    CustomObjectRepairBase, ConversationOperatorBase, ABC
):
    pass