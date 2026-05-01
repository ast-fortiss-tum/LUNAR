# Intent Definition and Policies

## 1. Intent Definitions

### 1.1 User Intent Set

| Intent          | Definition                                |
| --------------- | ----------------------------------------- |
| START           | Initial state of a conversation           |
| CHOICE          | User selects among presented options      |
| ADD_PREFERENCES | User provides constraints or preferences  |
| ASK             | User asks a question                      |
| CHANGE_OF_MIND  | User modifies a previous decision         |
| CONFIRMATION    | User confirms a system proposal           |
| REJECT          | User rejects a system proposal            |
| REJECT_CLARIFY  | User rejects with clarification           |
| REPEAT          | User requests repetition of prior content |
| STOP            | User terminates the interaction           |

---

### 1.2 Optimizable User Intent Subset

| Intent          |
| --------------- |
| CHOICE          |
| ADD_PREFERENCES |
| ASK             |
| CHANGE_OF_MIND  |
| CONFIRMATION    |
| REJECT          |
| REJECT_CLARIFY  |
| REPEAT          |
| STOP            |

---

### 1.3 System Intent Set

| Intent                    | Definition                               |
| ------------------------- | ---------------------------------------- |
| INFORM                    | Provide information                      |
| INFORM_AND_FOLLOWUP       | Provide information with response prompt |
| CHOICE                    | Present options                          |
| CHOICE_AND_FOLLOWUP       | Present options with prompt              |
| CONFIRMATION              | Request confirmation                     |
| CONFIRMATION_AND_FOLLOWUP | Confirmation with follow-up prompt       |
| CLARIFY                   | Request clarification                    |
| REJECT                    | Reject request                           |
| REJECT_AND_FOLLOWUP       | Reject with prompt                       |
| FAILURE                   | System failure state                     |
| MISC                      | Unclassified or fallback behavior        |

---

### 1.4 User Intent ID Mapping

| Mapping         | Description                     |
| --------------- | ------------------------------- |
| UserIntent → ID | Deterministic enumeration index |
| ID → UserIntent | Reverse lookup mapping          |

---

## 2. Policy Layer

## 2.1 System → User Transition Policy

| System Intent             | Allowed User Intents                                                             |
| ------------------------- | -------------------------------------------------------------------------------- |
| INFORM                    | ADD_PREFERENCES, ASK, CHANGE_OF_MIND, REJECT, REJECT_CLARIFY, STOP               |
| INFORM_AND_FOLLOWUP       | CHANGE_OF_MIND, ADD_PREFERENCES, CONFIRMATION, REJECT, REJECT_CLARIFY, ASK, STOP |
| CHOICE                    | CHOICE, ASK, ADD_PREFERENCES, CHANGE_OF_MIND, REJECT, REJECT_CLARIFY, STOP       |
| CHOICE_AND_FOLLOWUP       | CHOICE, ADD_PREFERENCES, CHANGE_OF_MIND, REJECT, REJECT_CLARIFY, STOP            |
| CONFIRMATION              | CHANGE_OF_MIND, ASK, ADD_PREFERENCES, STOP                                       |
| CONFIRMATION_AND_FOLLOWUP | CHANGE_OF_MIND, ADD_PREFERENCES, CONFIRMATION, STOP                              |
| CLARIFY                   | ADD_PREFERENCES, ASK, STOP                                                       |
| REJECT                    | CHANGE_OF_MIND, ASK, STOP                                                        |
| REJECT_AND_FOLLOWUP       | CHANGE_OF_MIND, ASK, CONFIRMATION, STOP                                          |
| FAILURE                   | CHANGE_OF_MIND, REPEAT, STOP                                                     |
| MISC                      | CHANGE_OF_MIND, ASK, REJECT, REJECT_CLARIFY, STOP                                |

---

## 2.2 Pre-Confirmation Policy

| Rule                   | Constraint                                                           |
| ---------------------- | -------------------------------------------------------------------- |
| Activation condition   | CONFIRMATION allowed only after ≥2 distinct pre-confirmation intents |
| Pre-confirmation set   | CHANGE_OF_MIND, REJECT, REJECT_CLARIFY, ADD_PREFERENCES              |
| Post-confirmation rule | All pre-confirmation intents are disabled after CONFIRMATION         |

---

## 2.3 Structural Constraints

| Constraint         | Rule                                                          |
| ------------------ | ------------------------------------------------------------- |
| START usage        | Only valid at first turn                                      |
| STOP usage         | Disabled until minimum dialogue length is reached             |
| REPEAT usage       | Limited by repetition budget                                  |
| Repeat suppression | Previous intent may be excluded if repetition is disallowed   |
| Feature dependency | ADD_PREFERENCES and REJECT_CLARIFY require available features |

---

## 2.4 Feature Availability Constraint

| Condition                  | Effect                                                     |
| -------------------------- | ---------------------------------------------------------- |
| No unused content features | Removes ADD_PREFERENCES and REJECT_CLARIFY from candidates |

---

## 2.5 Post-Confirmation Constraint

| Condition        | Effect                                               |
| ---------------- | ---------------------------------------------------- |
| confirmed = True | Blocks all pre-confirmation intents and CONFIRMATION |

---

## 2.6 Selection Policy (Abstract)

Intent selection is defined as a constrained stochastic process:

1. Filter candidate intents using policy constraints
2. Assign weights via priority function
3. Normalize weights into probability distribution
4. Sample intent from resulting distribution

---

## 3. Summary

The system implements a structured intent control framework combining:

* A formal intent ontology (user and system intents)
* Transition constraints from system outputs to user responses
* Temporal and state-dependent policy restrictions
* Pre-confirmation gating mechanism
* Stochastic intent selection under weighted priorities

The resulting model defines a constrained probabilistic policy over dialogue transitions.
