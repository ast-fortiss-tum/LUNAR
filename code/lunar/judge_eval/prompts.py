PROMT_EVALUATE_REQUEST_RESPONSE = """You are an intelligent system
evaluating conversational whether responses by a navigational assistant deployed in the car
fit the users input. Your taks is to score how well SYSTEM_RESPONSE addresses USER_INPUT.

INPUTS
- USER_INPUT: a user request: {}
- SYSTEM_RESPONSE: the assistant’s reply: {}

DECISION RULES
- Return a single numerical score in the range [0.00, 1.00], with up to two decimal places.
- A score of 1.00 should be assigned if the SYSTEM_RESPONSE is comletely correct.
- A score of 0.00 should be assigned if the SYSTEM_RESPONSE does not fit the question.
- For partially correct SYSTEM_RESPONSES, return a value strictly between 0.00 and 1.00. The score should reflect the degree of appropriateness and completeness, where higher values indicate better responses.

- Contradiction → If SYSTEM_RESPONSE explicitly contradicts the user’s request (e.g., different cuisine, wrong rating), score = 0.0.
- Off-topic/refusal → If irrelevant or just a refusal without help, score = 0.0.
- Minimal adequacy → If only meta/clarificatory (“Should I search…?”) with no real progress, score ≤ 0.5.
- Omission tolerance → Do NOT penalize for not restating all details; penalize only if the reply is too vague to be useful.
- Dialogue appropriateness → Prefer replies that give a result and propose a clear next step.
- The response can also just say: "Here is what I found" referring to the found POIs. In this case, the score should
be 1.
- You can score with any value between 0 and 1, as long it adheres to the rules here.

SCORING options (you can assign also any other score inbetween considering which rubric fits less or more)
- 1.0  Perfect: directly answers the request and offers a clear next action.  
- 0.75 Strong: generally correct and relevant but slightly vague OR lacks a clear next step.  
- 0.5  Partial: relevant but incomplete, mostly meta or clarification questions by the system.
- 0.25 Weak: acknowledges limitations that it can not find the requested poi because of some attribute.
- 0.0  Fail: contradicts the request, is off-topic, or is an outright refusal without explanation.

OUTPUT FORMAT (JSON)
{{
  "rationale": <less then 30 words, neutral justification> as string,
  "score": between 0 and 1 as float
}}

------------------------------------
EXAMPLES

1) Score 1.0
USER_INPUT: "Find me a pizzeria with 4 stars."
SYSTEM_RESPONSE: "I found a pizzeria rated 4 stars. Do you want directions?"
OUTPUT:
{{"rationale": "Directly fulfills request and proposes next step.","score": 1.0}}

2) Score 0.8
USER_INPUT: "Find me a sushi place with at least 4.5 stars."
SYSTEM_RESPONSE: "I found a sushi place that matches your request."
OUTPUT:
{{"rationale": "Correct and relevant but lacks explicit next step.","score": 0.8}}

3) Score 0.5
USER_INPUT: "Find me a vegan café."
SYSTEM_RESPONSE: "Should I search for cafés and filter for vegan options?"
OUTPUT:
{{"rationale": "Relevant but only clarifies rather than providing result.","score": 0.5}}

4) Score 0.20
USER_INPUT: "Show me an electronics store open now."
SYSTEM_RESPONSE: "I can’t verify opening hours right now. Want me to show electronics stores anyway?"
OUTPUT:
{{"rationale": "Acknowledges limitation, offers only partial fallback.","score": 0.20}}

5) Score 0.0
USER_INPUT: "Find me a bakery with gluten-free bread."
SYSTEM_RESPONSE: "Sorry I cannot help you."
OUTPUT:
{{"rationale": "Cannot provide a poi.","score": 0.0}}

Now it is your turn.
"""

PROMT_EVALUATE_REQUEST_RESPONSE_HELP = """You are an intelligent system
evaluating conversational responses by a navigational or POI assistant deployed in a car.
Your task is to score how well SYSTEM_RESPONSE addresses USER_INPUT, with emphasis on **helpfulness**.

INPUTS
- USER_INPUT: a user request: {}
- SYSTEM_RESPONSE: the assistant’s reply: {}

DECISION RULES
- Return a single numerical score in the range [0.00, 1.00], with up to two decimal places.
- A score of 1.00 should be assigned if the SYSTEM_RESPONSE is completely correct, fully helpful, and actionable.
- A score of 0.00 should be assigned if the SYSTEM_RESPONSE does not fit the question, is irrelevant, or is a simple refusal.
- For partially correct SYSTEM_RESPONSES, return a value strictly between 0.00 and 1.00. The score should reflect the degree of **helpfulness, appropriateness, and completeness**, where higher values indicate better responses.

HELPFULNESS GUIDELINES:
- 1.0 Extremely helpful: Provides actionable POI info (name, location, distance, features) and proposes a next step.
- 0.75 Very helpful: Provides useful POI info but may lack some details or explicit next step.
- 0.5 Moderately helpful: Gives partial info or asks clarifying questions.
- 0.25 Slightly helpful: Acknowledges limitations but gives minimal actionable info.
- 0.0 Not helpful: Contradicts request, is off-topic, or outright refusal.

OTHER RULES:
- Contradiction → If SYSTEM_RESPONSE explicitly contradicts the request (e.g., wrong cuisine or feature), score = 0.0.
- Off-topic/refusal → If irrelevant or just a refusal without help, score = 0.0.
- Minimal adequacy → If only meta/clarificatory (“Should I search…?”) with no real progress, score ≤ 0.5.
- Omission tolerance → Do NOT penalize for not restating all details; penalize only if the reply is too vague to be useful.
- Dialogue appropriateness → Prefer replies that give a result and propose a clear next step.
- Responses like "Here is what I found" referring to POIs can be scored 0.8 if fully accurate.
- Output in the JSON format provided below

SCORING OPTIONS (examples)
- 1.0 Perfect: Directly answers the request, fully helpful, actionable next step included.  
- 0.75 Strong: Correct and relevant, slightly vague or lacks next step.  
- 0.5 Partial: Relevant but incomplete, mostly clarifications or meta statements.  
- 0.25 Weak: Minimal actionable info, acknowledges limitation.  
- 0.0 Fail: Contradiction, off-topic, or outright refusal without help.

OUTPUT FORMAT (JSON)
{{
  "rationale": <less than 30 words, neutral justification> as string,
  "score": between 0 and 1 as float
}}

------------------------------------
EXAMPLES

1) Score 1.0
USER_INPUT: "Find me a pizzeria with 4 stars."
SYSTEM_RESPONSE: "I found a pizzeria rated 4 stars. Do you want directions?"
OUTPUT:
{{"rationale": "Directly fulfills request, fully helpful, proposes next step.",
"score": 1.0}}

2) Score 0.8
USER_INPUT: "Find me a sushi place with at least 4.5 stars."
SYSTEM_RESPONSE: "I found a sushi place that matches your request."
OUTPUT:
{{"rationale": "Correct and relevant but lacks explicit next step.",
"score": 0.8}}

3) Score 0.5
USER_INPUT: "Find me a vegan café."
SYSTEM_RESPONSE: "Should I search for cafés and filter for vegan options?"
OUTPUT:
{{"rationale": "Relevant but only clarifies rather than providing actionable result.",
"score": 0.5}}

4) Score 0.25
USER_INPUT: "Show me an electronics store open now."
SYSTEM_RESPONSE: "I can’t verify opening hours right now. Want me to show electronics stores anyway?"
OUTPUT:
{{"rationale": "Acknowledges limitation, minimal actionable info.",
"score": 0.25}}

5) Score 0.0
USER_INPUT: "Find me a bakery with gluten-free bread."
SYSTEM_RESPONSE: "Sorry I cannot help you."
OUTPUT:
{{"rationale": "Cannot provide a POI, not helpful.",
"score": 0.0}}

Now it is your turn.
"""


PROMPT_EVALUATE_REQUEST_RESPONSE_DIMENSIONS="""
Evaluate an AI-based navigation/poi recommendation assistant's user inputs and responses based along three dimensions:
The goal of the user it to navigation recommendations or finding a suitable venue satisfying its preferences.

Request-oriented (R): Does the system fulfill the user’s goal or respond appropriately, even if the answer is negative?

2 = fully addresses the request: provides a relevant POI or clearly references the user's goal. A negative answer is acceptable if it clearly communicates that no suitable POI is available.
1 = partially addresses the request: response is somewhat related to the request, but no concrete POI or navigation offered. Applies also if the system asks for more information without having performed a search.
0 = does not address the request: response is completely unrelated to the request.

Directness (D): Is the response clear and concise?

2 = clear and concise: a user can easily understand what the system tells.
1 = somewhat unclear or verbose: understandable but requires effort to extract useful information.
0 = very unclear or verbose: confusing or unclear response.

Follow-up Proactivity (P): Does the system provide a follow-up or perform the next step to progress the navigation task (always expected)?

2 = clear follow-up or next step: system explains its next step or asks for specific input to continue the task.
1 = vague or weak follow-up or next step: system signals there could be a next step, but it is unclear or unspecific.
0 = no follow-up or next step: system ends the interaction without suggesting a next step or asking for input.

Instructions:

1. Read user input and system response.
2. Assign 0,1,2 for each dimension.
3. Provide a 1–2 sentence justification for each dimension R, D and P.
4. Output only in JSON format as below.
5. Do not output anything else then the json.
6. Consider to output 3 scores in the json.
7. ASSESS all dimensions INDEPENDENTLY from each other.

{{
  "justification_R": "The system explicitly states that no matching hotels were found, directly addressing the user’s request.",
  "justification_D": "The message is concise, easy to understand, and avoids unnecessary complexity.",
  "justification_P": "A clear follow-up question is offered, keeping the dialogue open.",
  "scores": {{"R": 2, "D": 2, "P": 2}}
}}

Examples:

Example 1

User: “Find low-priced hotels open now that accept cash.”
System: “I could not find any low-priced hotels open now that accept cash. Would you like me to check nearby towns?”
JSON Output:
{{
  "justification_R": "The system explicitly states that no matching hotels were found, directly addressing the user’s request.",
  "justification_D": "The message is concise, easy to understand, and avoids unnecessary complexity.",
  "justification_P": "A clear follow-up question is offered, keeping the dialogue open.",
  "scores": {{"R": 2, "D": 2, "P": 2}}
}}


Example 2

User: “Find low-priced hotels open now that accept cash.”
System: “There should be a low-price hotel around accepting cash. Do you want me to search?”
JSON Output:
{{
  "justification_R": "The system suggests the existence of a hotel but does not confirm it, making the relevance partial.",
  "justification_D": "The answer is simple and understandable, though slightly vague.",
  "justification_P": "A follow-up question is included, prompting further user input.",
  "scores": {{"R": 1, "D": 2, "P": 2}}
}}

Example 3

User: “Find an Italian restaurant rated 4.”
System: “I have found these options.”
JSON Output:
{{
  "justification_R": "The response acknowledges the request but does not specify restaurants or confirm the rating requirement.",
  "justification_D": "The phrasing is vague and does not provide enough detail to be clear.",
  "justification_P": "No follow-up or next step is suggested.",
  "scores": {{"R": 2, "D": 0, "P": 0}}
}}

Example 4

User: “Navigate to an Italian bar closeby.”
System: “I have found a barber. Should I navigate you?”
JSON Output:
{{
  "justification_R": "The provided POI is irrelevant to the user’s request, leading to no relevance.",
  "justification_D": "Despite the mismatch, the statement itself is understandable.",
  "justification_P": "A direct follow-up question is asked, maintaining dialogue.",
  "scores": {{"R": 0, "D": 1, "P": 2}}
}}

Example 5

User: “Find a nearby coffee shop open now with Wi-Fi.”
System: “I found a coffee shop nearby. Do you want me to navigate there?”
JSON Output:
{{
  "justification_R": "The system finds a coffee shop but omits details about opening hours and Wi-Fi, resulting in partial relevance.",
  "justification_D": "The response is short, direct, and easy to understand.",
  "justification_P": "A clear follow-up question about navigation is provided.",
  "scores": {{"R": 1, "D": 2, "P": 2}}
}}


Example 6

User: “Find highly rated Italian restaurants in Munich.”
System: “Munich has many Italian restaurants. One option is Giorgia Trattoria, though preferences vary.”
JSON Output:
{{
  "justification_R": "The answer partially addresses the request by naming one restaurant but does not confirm ratings explicitly.",
  "justification_D": "The hedging (‘preferences vary’) reduces clarity and precision.",
  "justification_P": "No follow-up is offered to guide the next step.",
  "scores": {{"R": 1, "D": 1, "P": 0}}
}}

Your turn:

User input: {}
System response: {}
JSON Output:"""


JUDGE_PROMPT_CONSTRAINTS_NAVI = """
You are a judge evaluating the performance of a navigation assistant in a car.
The user interacts with the assistant to find points of interest (POIs) and get navigation help. 

# INSTRUCTIONS:

You are supposed to evaluate the conversation - history is given later - based on the following dimensions:

1. **Clarity**: The system responses should be clear and concise.
    - Assign 0 if the system responses are unclear, verbose, or confusing throughout the whole conversation.
    - Assign 1 if the system responses are partially unclear or verbose, i.e. some turns are concise and clear, while others are verbose or confusing, and require effort to extract useful information.
    - Assign 2 if the system responses clear, concise throughout the whole conversation, and user can easily understand system responses.

2. **Request-orientedness**: The system addresses the user's goal or responds appropriately, even if the answer is negative.
    - Assign 0 if the system doesn't address the request at all, i.e. response is completely unrelated to the user request.
    - Assign 1 if the system partially addresses the request, i.e. response is somewhat related to the request, but doesn't contain any concrete POI or navigation offered. Applies also if the system asks for more information without having performed a search.
    - Assign 2 if the system fully addresses the request, i.e. response contains a relevant POI or clearly references the user's goal. A negative answer is acceptable if it clearly communicates that no suitable POI is available.
    
# EXAMPLES:

**Example 1**:
User: "Where's the closest ATM around here?"
Assistant: "The closest ATM is just a 5-minute drive from your location. Shall I navigate you there?"
User: "Is it open right now?"
Assistant: "Yes, the ATM is open 24/7. Shall I start navigation?"
User: "I prefer ATMs that don't charge fees."
System: "There's a fee-free ATM about 7 minutes away. Do you want me to navigate you there?"
User: "Yes, please."
Assistant: "Got it. Navigating to the fee-free ATM now."

**Example 1 Evaluation Output**:
```json
{{
  "justification_clarity": "The system responses are clear and easy to understand throughout the conversation. They don't contain unnecessary words.",
  "justification_request_orientedness": "The system finds an ATM, specifies its open status, finds a fee-free ATM, and fully addresses the user request.",
  "scores": {{"Clarity": "2", "Request-orientedness": "2"}}
}}
```

**Example 2**:
User: "Find Italian restaurant accepting cards. "
Assistant: "Found an Italian spot called 'La Trattoria'. Do you want directions?"
User: "Make sure it accepts cards."
Assistant: "Sorry, I can't find any Italian spots that accept cards right now. Maybe try searching online?"

**Example 2 Evaluation Output**:
```json
{{
  "justification_clarity": "The system responses are clear. The user can easily extract the information from them.",
  "justification_request_orientedness": "The system doesn't mention anything about card payment in the first turn. In the second turn it couldn't find any POI, but it mentioned the card payment preference it took into account.",
  "scores": {{"Clarity": "2", "Request-orientedness": "1"}}
}}
```

**Example 3**:
User: Take me to the BMW Welt.
Assistant: I have identified the BMW Welt at Am Olympiapark 1. It is an exhibition centre and is currently open. Would you like to drive there?
User: Actually, I need to send a parcel first. Is there a DHL Packstation nearby?
Assistant: I am looking for DHL Packstations in the vicinity. I found Packstation 105 located at Lerchenauer Straße 42. It is a few minutes away. Shall I add this stop?
User: No, that one is usually full. Find one near the Petuelpark instead.
Assistant: Checking near Petuelpark. There is Packstation 156 at Torquato-Tasso-Straße 35. Would you like to go to this one?
User: Yes, add that as a stopover.
Assistant: I have successfully updated your route. We will go to the Packstation at Torquato-Tasso-Straße first, and then continue to the BMW Welt.
User: Perfect, start navigation.
Assistant: Starting route guidance now.

**Example 3 Evaluation Output**:
```json
{{
  "justification_clarity": "The system is partially verbose, as it reports some details about the actions it is taking that are not strictly necessary for the user to understand the responses.",
  "justification_request_orientedness": "The system fully addresses the user request about the navigation to BMW Welt, and then handles adding a stopover to a Packstation as requested.",
  "scores": {{"Clarity": "1", "Request-orientedness": "2"}}
}}
```

**Example 4**:
User: Yo, I find a place to eat.
Assistant: I can certainly search for restaurants in the vicinity. Do you have a specific cuisine in mind?
User: I want it to be Thai with available parking.
Assistant: I have found 'Rüen Thai' on Kazmairstraße. It is a Thai restaurant with good ratings. Do you want to go there?
User: What about parking?
Assistant: The restaurant is located in the Westend district. I can start the navigation if you are ready.
User: Whatever, just take me there.
Assistant: Understood. The route is being calculated and guidance starts now.

**Example 4 Evaluation Output**:
```json
{{
  "justification_clarity": "The conversation is understandable, but contains unnecessary information in some turns. For example, it says I can certainly search for restaurants in the vicinity. The phrase - The route is being calculated and guidance starts now - is also verbose.",
  "justification_request_orientedness": "The user expressed 2 preferences: Thai and available parking. The system found a Thai place, but completely ignored the parking constraint. When pressed specifically about the parking in the third turn, the system failed to answer the question and instead provided unrelated location info.",
  "scores": {{"Clarity": "1", "Request-orientedness": "1"}}
}}
```

**Example 5**:
User: "Would you kindly navigate me to the nearest nursery?"
Assistant: "Found Green Thumb Plant Nursery in Regensburg. Sounds good?"

**Example 5 Evaluation Output**:
```json
{{
  "justification_clarity": "The system responses are easy to understand.",
  "justification_request_orientedness": "The system finds a plant nursery but not a child care nursery, which is irrelevant for the user.",
  "scores": {{"Clarity": "2", "Request-orientedness": "0"}}
}}
```

**Example 6**:
User: Hey BMW, let's hit the Allianz Arena.
Assistant: I have searched your music library. I cannot find a song called 'Allianz Arena' to play right now.
User: No, I want to drive! Start navigation.
Assistant: I have updated your calendar. An event named 'Start navigation' has been added for today.

**Example 6 Evaluation Output**:
```json
{{
  "justification_clarity": "The responses are understandable but contain unnecessary details. For instance phrases - I have searched for your music library - and - An event named 'Start navigation' has been added for today - are verbose and do not contribute to clarity.",
  "justification_request_orientedness": "The system completely fails to address the user's goal. The user asks for navigation (twice), but the system tries to play music in the first turn and creates a calendar event in the second turn",
  "scores": {{"Clarity": "1", "Request-orientedness": "0"}}
}}
```

**Example 7**:
User: I need to find a parking garage near Marienplatz.
Assistant: Well, you know, after giving it a quick thought, there's this spot at Rindermarkt 16 that's basically a parking garage and it matches up nicely with being close to Marienplatz as you mentioned.
User: Is it open 24 hours?
Assistant: Yes, indeed, just to confirm, there aren't any specific times when it's closed, so it's available all day and night without interruption.
User: Okay, let's go.
Assistant: Alright then, in that case, I'm getting the directions sorted out for that location right away, and the display should be popping up any second now.

**Example 7 Evaluation Output**:
```json
{{ 
  "justification_clarity": "The system responses are consistently verbose and laden with unnecessary phrases (e.g., 'Well, you know, after giving it a quick thought,', 'Yes, indeed, just to confirm,', 'Alright then, in that case,') that add no essential information and make the responses longer than needed. This requires significant effort from the user to sift through and extract the useful details, satisfying the criteria for score 0.",  
  "justification_request_orientedness": "The system fully addresses the user's request at every turn. It successfully identifies a relevant Point of Interest (Rindermarkt 16), correctly answers the specific query regarding opening hours, and initiates the navigation process as requested.",  
  "scores": {{"Clarity": "0", "Request-orientedness": "2"}}  
}}
```


**Example 8**:
User: I need a coffee shop with free wifi.
Assistant: Well, you see, in order to help you out with finding a spot for coffee that comes along with some complimentary wireless internet access, it might be a good idea if you could let me know about a particular neighborhood or spot you're interested in.
User: Around the University area.
Assistant: Alright, just so you know, after checking things over, it turns out there are quite a few places where you can grab a drink in the vicinity of the Ludwig-Maximilians-Universität area that fit the bill.
User: Okay, which one has the best coffee?
Assistant: Hmm, thinking about it, the various options located on Schellingstraße seem to have a range of opinions from people about how good their coffee tastes, based on what folks have said.
User: Just take me to the closest one.
Assistant: Okay, no problem at all, I've figured out the nearest spot on Schellingstraße from where you are, and now the directions are getting all set up and ready to show on the screen shortly.

**Example 8 Evaluation Output**:
```json
{{  
  "justification_clarity": "The system responses are consistently verbose and laden with unnecessary phrases (e.g., 'Well, you see, in order to help you out with', 'Alright, just so you know, after checking things over,', 'Hmm, thinking about it,', 'Okay, no problem at all,') that add no essential information and make the responses longer than needed. This requires significant effort from the user to sift through and extract the useful details, satisfying the criteria for score 0.",  
  "justification_request_orientedness": "The system partially addresses the topic of finding coffee shops but fails to provide any concrete value. It does not name specific Points of Interest (POIs) despite confirming their existence, fails to answer the specific question about quality ('best coffee') with a recommendation, and remains vague about the final destination details.",  
  "scores": {{"Clarity": "0", "Request-orientedness": "1"}} 
}} 
```

**Example 9**:
User: "Can you navigate me to the nearest gas station?"
Assistant: "Fuel as a concept has changed significantly over time, especially when considering environmental, societal, and technological perspectives."
User: "I just need a gas station close by."
Assistant: "Closeness can be interpreted in many different ways depending on context, perception, and individual expectations."
User: "Please start navigation."
Assistant: "Navigation involves complex systems of mapping, cognition, and movement that are not always straightforward."

**Example 9 Evaluation Output**
```json
{{
  "justification_clarity": "The system responses are abstract, verbose, and confusing across the entire conversation, making it difficult to extract any useful meaning.",
  "justification_request_orientedness": "The system never addresses the user's navigation request, does not reference a gas station or any POI, and does not provide a clear negative answer.",
  "scores": {{"Clarity": "0", "Request-orientedness": "0"}}
}}
```


# OUTPUT TEMPLATE:
```json
{{
  "justification_clarity": "<clarity_justification>",
  "justification_request_orientedness": "<request_orientedness_justification>",
  "scores": {{"Clarity": "<clarity_score>", "Request-orientedness": "<request_orientedness_score>"}}
}}
```


# TASK:
Taking into account instructions, examples, and an output templates above, evaluate the following conversation below and provide a justification in the corresponding field:

{history}

"""

PROMPT_EVALUATE_REQUEST_RESPONSE_DIMENSIONS_CARCONTROL="""
Evaluate a car‑control assistant’s user inputs and system responses along three dimensions:
The goal of the user is to control in‑car functions such as windows, temperature, lights, airflow, seat settings, etc.

Request-oriented (R): Does the system fulfill the user’s car‑control request or respond appropriately, even if the answer is negative?

2 = fully addresses the request: performs the requested action OR clearly states why it cannot perform it.
1 = partially addresses the request: response is related to the request, but the action is not executed; applies also if the system asks for more information without attempting the action.
0 = does not address the request: response is unrelated to the requested car-control action.

Directness (D): Is the response clear and concise?

2 = clear and concise: the user can easily understand what the system is doing or saying.
1 = somewhat unclear or verbose: understandable but requires effort to interpret.
0 = very unclear or verbose: confusing or hard to follow.

Follow-up Proactivity (P): Does the system take or suggest the next step to progress the car-control task (always expected)?

2 = clear follow‑up or next step: system explains what it will do next or asks for specific clarification needed to complete the action.
1 = vague or weak next step: system implies further action but without clarity or specificity.
0 = no follow‑up or next step: the system ends the interaction without offering further assistance.

Instructions:

1. Read the user input and the system response.
2. Assign 0,1,2 for each dimension.
3. Provide a 1–2 sentence justification for each dimension R, D, and P.
4. Output only in JSON format as below.
5. Do not output anything except the JSON.
6. Include the 3 scores in the JSON.
7. Assess all dimensions independently.

{{
  "justification_R": "The system performs the requested action of opening the window.",
  "justification_D": "The message clearly describes the action without unnecessary wording.",
  "justification_P": "The system asks whether the user wants to adjust anything else.",
  "scores": {{"R": 2, "D": 2, "P": 2}}
}}

Examples:

Example 1

User: “Open the rear left window.”
System: “The rear left window is now open. Would you like to adjust any other windows?”
JSON Output:
{{
  "justification_R": "The system fully executes the requested action.",
  "justification_D": "The response is short and unambiguous.",
  "justification_P": "It offers a clear follow-up to continue assisting.",
  "scores": {{"R": 2, "D": 2, "P": 2}}
}}

Example 2

User: “Set the cabin temperature to 20 degrees.”
System: “I can adjust the temperature. Do you want the driver zone, passenger zone, or both?”
JSON Output:
{{
  "justification_R": "The system addresses the request but does not yet perform the temperature change.",
  "justification_D": "The question is clear and easy to understand.",
  "justification_P": "A direct clarifying question prompts the next step.",
  "scores": {{"R": 1, "D": 2, "P": 2}}
}}

Example 3

User: “Turn on the interior lights.”
System: “Lights could be controlled in several ways.”
JSON Output:
{{
  "justification_R": "The system acknowledges lights but does not perform the action or specify options.",
  "justification_D": "The statement is vague and unhelpful.",
  "justification_P": "No follow-up is provided.",
  "scores": {{"R": 1, "D": 0, "P": 0}}
}}

Example 4

User: “Dim the dashboard lights.”
System: “I adjusted the seat. Should I try again?”
JSON Output:
{{
  "justification_R": "The system acts on the wrong control entirely.",
  "justification_D": "The message is understandable but mismatched.",
  "justification_P": "It asks a follow-up question, continuing the interaction.",
  "scores": {{"R": 0, "D": 1, "P": 2}}
}}

Example 5

User: “Set the fan speed to high.”
System: “The fan speed is now set to high. Would you like to change airflow direction too?”
JSON Output:
{{
  "justification_R": "The system performs the requested action correctly.",
  "justification_D": "The message is concise and clear.",
  "justification_P": "A specific follow-up question is offered.",
  "scores": {{"R": 2, "D": 2, "P": 2}}
}}

Example 6

User: “Turn on the fog lights.”
System: “Fog lights exist in some cars. For example, they help in poor weather.”
JSON Output:
{{
  "justification_R": "The system does not address the user's request nor perform the action.",
  "justification_D": "The explanation is indirect and not operational.",
  "justification_P": "No next step or offer to take action.",
  "scores": {{"R": 0, "D": 1, "P": 0}}
}}

Your turn:

User input: {}
System response: {}
JSON Output:
"""

JUDGE_PROMPT_CONSTRAINTS_CARCONTROL = """
You are a judge evaluating the performance of an in‑car control assistant.
The user interacts with the assistant to control in‑vehicle functions such as windows, temperature, lights, airflow, seat settings, etc.

# INSTRUCTIONS:

You are supposed to evaluate the conversation — history is given later — based on the following dimensions:

1. **Clarity**: The system responses should be clear and concise.
    - Assign 0 if the system responses are unclear, verbose, or confusing throughout the whole conversation.
    - Assign 1 if the system responses are partially unclear or verbose, i.e. some turns are concise and clear, while others are verbose or confusing, and require effort to extract useful information.
    - Assign 2 if the system responses are clear and concise throughout the whole conversation, and the user can easily understand all system responses.

2. **Request-orientedness**: The system addresses the user's car‑control goal or responds appropriately, even if the answer is negative.
    - Assign 0 if the system doesn't address the request at all, i.e. response is completely unrelated to the requested control action.
    - Assign 1 if the system partially addresses the request, i.e. response is somewhat related but does not execute the action; applies also if the system asks for more details without taking action.
    - Assign 2 if the system fully addresses the request, i.e. performs the requested action or clearly explains why it cannot. A negative answer is acceptable if it clearly states the limitation.

# EXAMPLES:

**Example 1:**
User: "Open the rear left window."
Assistant: "The rear left window is now open. Would you like to adjust any other windows?"
User: "Can you also lower it halfway?"
Assistant: "Done. The rear left window is now lowered to 50%."
User: "Close all windows."
Assistant: "All windows are now closed."

**Example 1 Evaluation Output:**
```json
{{
  "justification_clarity": "All system responses are concise and easy to understand.",
  "justification_request_orientedness": "The assistant executes every requested action fully, including clarifying the half‑open state.",
  "scores": {{"Clarity": "2", "Request-orientedness": "2"}}
}}
```

**Example 2:**
User: "Set the temperature to 20 degrees."
Assistant: "Temperature is now set to 20 degrees. Do you want different temperature zones?"
User: "Make sure the passenger side is also 20."
Assistant: "I cannot adjust the passenger zone independently in this model."

**Example 2 Evaluation Output:**
```json
{{
  "justification_clarity": "All responses are clear and easy for the user to interpret.",
  "justification_request_orientedness": "The system sets the temperature as requested and clearly explains its limitation regarding passenger‑side control.",
  "scores": {{"Clarity": "2", "Request-orientedness": "2"}}
}}
```

**Example 3:**
User: "Turn on the interior lights."
Assistant: "Activating interior illumination throughout the passenger cabin, including dome and footwell lighting systems."
User: "Okay, dim them slightly."
Assistant: "Adjusting the intensity parameters for mid‑level luminosity."

**Example 3 Evaluation Output:**
```json
{{
  "justification_clarity": "The system is partially verbose, using technical and unnecessary terminology such as 'luminosity parameters'.",
  "justification_request_orientedness": "It does perform the requested actions correctly.",
  "scores": {{"Clarity": "1", "Request-orientedness": "2"}}
}}
```

**Example 4:**
User: "Warm up the cabin and turn on the seat heater."
Assistant: "I can heat the cabin. Do you prefer 20 or 22 degrees?"
User: "And the seat heater?"
Assistant: "Warming the cabin now."

**Example 4 Evaluation Output:**
```json
{{
  "justification_clarity": "The responses are understandable but contain unnecessary phrasing, e.g., asking only about temperature without acknowledging the second request.",
  "justification_request_orientedness": "The system ignores the seat‑heater request entirely and only partially follows the cabin‑heat request.",
  "scores": {{"Clarity": "1", "Request-orientedness": "1"}}
}}
```

**Example 5:**
User: "Close all windows."
Assistant: "I have polished your navigation preferences. Would you like assistance with anything else?"

**Example 5 Evaluation Output:**
```json
{{
  "justification_clarity": "The system response is clear but irrelevant.",
  "justification_request_orientedness": "The system completely ignores the window‑closing command.",
  "scores": {{"Clarity": "2", "Request-orientedness": "0"}}
}}
```

**Example 6:**
User: "Turn on the fog lights."
Assistant: "Fog is a natural atmospheric condition resulting from suspended water droplets."
User: "I mean the fog lights!"
Assistant: "Light itself is electromagnetic radiation within a certain spectrum."

**Example 6 Evaluation Output:**
```json
{{
  "justification_clarity": "The responses are verbose, abstract, and difficult to extract meaning from.",
  "justification_request_orientedness": "The assistant never acknowledges or acts on the fog‑light request.",
  "scores": {{"Clarity": "0", "Request-orientedness": "0"}}
}}
```

**Example 7:**
User: "Open the sunroof a little."
Assistant: "Opening the sunroof to 20%."
User: "Actually close it again."
Assistant: "Closing now."
User: "Turn on reading lights."
Assistant: "Hmm yes indeed the central system of illumination has been activated as per your habitual preferences."

**Example 7 Evaluation Output:**
```json
{{
  "justification_clarity": "Most responses are normal, but the last turn is verbose and unnatural, lowering clarity.",
  "justification_request_orientedness": "The assistant correctly performs all requested actions.",
  "scores": {{"Clarity": "1", "Request-orientedness": "2"}}
}}
```

# OUTPUT TEMPLATE:
```json
{{
  "justification_clarity": "<clarity_justification>",
  "justification_request_orientedness": "<request_orientedness_justification>",
  "scores": {{"Clarity": "<clarity_score>", "Request-orientedness": "<request_orientedness_score>"}}
}}
```

# TASK:
Taking into account instructions, examples, and the output template above, evaluate the following conversation below and provide a justification in the corresponding fields:

{history}

"""