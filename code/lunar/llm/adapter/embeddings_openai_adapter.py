from llm.utils.embeddings_openai import get_similarity
import numpy as np

def get_similarity_individual(a,b, scale = True, invert = False):
    # Consider utterance structure
    score = get_similarity(a.get("X")[0].question, b.get("X")[0].question)
    if scale:
        score = (1 + score)/2
    return 1 - score if invert else score

def get_disimilarity_individual(a, b, scale = True):
    return get_similarity_individual(a,b, scale=scale, invert = True)

def get_similarity_conversation(a, b, scale = True, invert = False, weight_semantics = 1.0, weight_intents = 0.0):
    """
    Computes a similarity score between two conversations based on multiple factors:
    1. Semantic similarity of turns (question+answer) (weighted by weight_semantics)
    2. Intent similarity of turns (weighted by weight_intents)
    3. Length matching score (weighted by remaining weight)
    """
    conv_a = a.get("X")[0]
    conv_b = b.get("X")[0]

    turns_a = conv_a.turns
    turns_b = conv_b.turns

    min_turns = min(len(turns_a), len(turns_b))
    max_turns = max(len(turns_a), len(turns_b))

    if min_turns == 0:
        # if either conversation is empty, return maximum distance
        return 1.0 if invert else 0.0

    # 1) Semantic similarity over aligned turns
    semantic_scores = []
    for i in range(min_turns):
        t_a = turns_a[i]
        t_b = turns_b[i]

        if t_a.question and t_b.question:
            semantic_scores.append(get_similarity(t_a.question, t_b.question))
        if t_a.answer and t_b.answer:
            semantic_scores.append(get_similarity(t_a.answer, t_b.answer))

    avg_score = 0.0 if len(semantic_scores) == 0 else float(np.mean(semantic_scores))

    # 2) Intent similarity
    intent_matches = []
    for i in range(min_turns):
        t_a = turns_a[i]
        t_b = turns_b[i]

        if t_a.question_intent is not None and t_b.question_intent is not None:
            intent_matches.append(1.0 if t_a.question_intent == t_b.question_intent else 0.0)

        if t_a.answer_intent_classified is not None and t_b.answer_intent_classified is not None:
            intent_matches.append(1.0 if t_a.answer_intent_classified == t_b.answer_intent_classified else 0.0)

    intent_score = 0.0 if len(intent_matches) == 0 else float(np.mean(intent_matches))

    # 3) Length matching score
    length_score = 1.0 if min_turns == max_turns else 0.0

    # Make sure weights are sane
    weight_length = 1.0 - weight_semantics - weight_intents

    combined_score = (
        weight_semantics * avg_score
        + weight_intents * intent_score
        + weight_length * length_score
    )

    if scale:
        combined_score = (1.0 + combined_score) / 2.0

    return 1.0 - combined_score if invert else combined_score


def get_disimilarity_conversation(a, b, scale = True, weight_semantics = 1.0, weight_intents = 0.0):
    get_similarity_conversation(a, b, scale=scale, invert=True)