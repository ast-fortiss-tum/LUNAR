

<h1 align="center">🌙 LUNAR: Automated Testing of Multi-Turn Conversations in Task-Oriented LLM Systems
</h1>

LUNAR is an automated testing framework for benchmarking multi-turn conversations in task-oriented LLM applications. It models the conversation space using an intent-based formalization and applies genetic optimization to systematically identify failure-inducing interaction sequences.

## Updates

<div style="background:#fff6d8;border:1px solid #dfc15d;border-left:6px solid #b98c00;border-radius:10px;padding:14px 16px;margin:20px 0;color:#2f2400;line-height:1.6;">
    <p style="margin:6px 0 10px 0;"><strong>Date:</strong> May 28, 2026</p>
    <ul><li> Threshold sensitivity resulte are added: <a href="additional/threshold_variation.pdf" style="color:#005a9e;">here</a>
    </li>
    </ul>
</div>

---

## Table of Contents

- [Approach](#approach)
- [Replication](#Replication)
- [Results](#results)
- [Supplementary Material](#supplementary-material)

---

## Approach

LUNAR combines following components:

- **Intent-Based Formalization**  
  The conversation space is represented as a structured set of user utterance intents and possible system-user intent transitions, enabling systematic exploration.

- **Genetic Optimization**  
  A search-based strategy is used to efficiently navigate the conversation space and identify failure-inducing dialogues.

- **Generative Turn Generation**  
  LLM-based utterance generation for human-like input generation.

- **Conversation Evaluation**  
  Turn-wise native numerical goal/results assessment + intent-aware conversation efficiency evaluation.

This combination allows LUNAR to explore the complex test space systems to identify failure revealing test inputs.

## Replication

The code with installation and replication instructions is provided [here](code/README.md).

## Results

The metrics results and shared data can be found [here](results/README.md).

## Supplementary Material

The supplementary material which includes further results and information about system internals can be found [here](supplementary/lunar_supplementary_material.pdf).
