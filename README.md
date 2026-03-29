

<h1 align="center">🌙 LUNAR: Automated Testing of Multi-Turn Conversations in Task-Oriented LLM Systems
</h1>

LUNAR is an automated testing framework for benchmarking multi-turn conversations in task-oriented LLM applications. It models the conversation space using an intent-based formalization and applies genetic optimization to systematically identify failure-inducing interaction sequences.

---

## Table of Contents

- [Overview](#overview)
- [Approach](#approach)
- [Installation](#installation)
- [Replication](#Replication)
- [Customization](#customization)
- [Results](#results)
- [Supplementary Material](#supplementary-material)

---

## Overview

Testing conversational systems is inherently challenging due to the combinatorial explosion of possible interaction paths. LUNAR addresses this challenge by providing a structured and automated approach to explore multi-turn conversations and uncover system weaknesses.

---

## Approach

LUNAR combines three key components:

- **Intent-Based Formalization**  
  The conversation space is represented as a structured set of user utterance intents and possible system-user intent transitions, enabling systematic exploration.

- **Genetic Optimization**  
  A search-based strategy is used to efficiently navigate the conversation space and identify failure-inducing dialogues.

- **Generative Turn Generation**  
  LLM-based utterance generation for human-like input generation.

This combination allows LUNAR to explore the complex test space systems to identify failure revealing test inputs.

---

## Installation

🚧*Content will be provided soon.*

## Replication

🚧*Content will be provided soon.*

## Customization

🚧*Content will be provided soon.*


## Results

The metrics results and shared data can be found [here](results).

## Supplementary Material

The supplementary material which includes further results and information about system internals can be found [here](supplementary/lunar_supplementary_material.pdf).
