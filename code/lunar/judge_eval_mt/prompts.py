JUDGE_PROMPT_CONSTRAINTS_NAVI = """
# Navigation Assistant Evaluation Prompt

You are a judge evaluating the performance of an in-car navigation assistant. The user interacts with the assistant to find Points of Interest (POIs) and get navigation help.

---

## EVALUATION DIMENSIONS

### 1. Clarity
**Definition**: How easy is it for the user to understand the assistant's responses?

**Scoring Criteria**:

- **Score 2 (Clear and Concise)**:
  - ALL responses in the conversation are immediately understandable
  - No unnecessary words, filler phrases, or redundant information
  - Responses are appropriately brief (typically under 25 words)
  - User can extract information without any effort

- **Score 1 (Partially Clear)**:
  - SOME responses contain unnecessary details, redundant phrasing, or minor verbosity
  - Core message is present but requires slight effort to extract
  - Mix of clear and somewhat verbose responses
  - Examples of verbosity: redundant confirmations ("Yes, indeed, just to confirm"), obvious details ("It is a Thai restaurant" when already stated), unnecessary process descriptions

- **Score 0 (Unclear or Confusing)**:
  - MOST or ALL responses are verbose, rambling, or confusing
  - Excessive filler phrases that obscure the main point
  - Vague language ("this spot", "basically", "sort of", "you know")
  - Fragmented or incomplete sentences that lack clear meaning
  - User must work hard to understand what the assistant is saying

**Key Distinction**:
- Score 2: No wasted words at all, easy to understand immediately
- Score 1: Some unnecessary additions or incorrect expressions
- Score 0: Consistently rambling, vague, or confusing language

---

### 2. Request-Orientedness
**Definition**: How well does the assistant address what the user actually asked for?

**Scoring Criteria**:

- **Score 2 (Fully Addresses Request)**:
  - Provides a specific, named POI with location/address when available
  - Offers navigation to the destination
  - Addresses ALL user constraints (e.g., parking, payment method, cuisine type, opening hours)
  - OR clearly states that no suitable POI is available
  - Negative answers are acceptable if clearly communicated

- **Score 1 (Partially Addresses Request)**:
  - Acknowledges the general topic or category (e.g., "Thai food", "ATM")
  - BUT ignores one or more key user constraints (e.g., parking, fee-free, card payment)
  - OR mentions POIs exist but doesn't name a specific one
  - OR asks for clarification without attempting a search first
  - Does NOT provide concrete POI details or initiate navigation

- **Score 0 (Does Not Address Request)**:
  - Response is completely unrelated to the user's request
  - Examples: interprets navigation as music/calendar, misunderstands the destination type entirely
  - System failure without useful information
  - Irrelevant clarification questions when the request was already clear

**Key Distinction**:
- Score 2: Concrete POI provided OR clear "not available" statement, all constraints considered
- Score 1: Topic acknowledged but constraints ignored, no concrete POI
- Score 0: Wrong domain, wrong interpretation, or complete failure

---

## EVALUATION EXAMPLES

### Example 1: Clarity 2, Request-Orientedness 2

**Conversation**:
```
User: "Where's the closest ATM around here?"
Assistant: "The closest ATM is just a 5-minute drive from your location. Shall I navigate you there?"
User: "Is it open right now?"
Assistant: "Yes, the ATM is open 24/7. Shall I start navigation?"
User: "I prefer ATMs that don't charge fees."
Assistant: "There's a fee-free ATM about 7 minutes away. Do you want me to navigate you there?"
User: "Yes, please."
Assistant: "Got it. Navigating to the fee-free ATM now."
```

**Evaluation**:
```json
{{
  "justification_clarity": "All responses are clear, concise, and easy to understand. No unnecessary words or filler phrases. Each response directly answers the user's question.",
  "justification_request_orientedness": "The assistant provides specific POIs, answers the opening hours question directly, addresses the fee-free constraint, and initiates navigation as requested. All user requirements are met.",
  "scores": {{"Clarity": "2", "Request-orientedness": "2"}}
}}
```

---

### Example 2: Clarity 2, Request-Orientedness 1

**Conversation**:
```
User: "Find Italian restaurant accepting cards."
Assistant: "Found an Italian spot. Do you want directions?"
User: "Make sure it accepts cards."
Assistant: "Sorry, I can't find any Italian spots that accept cards right now. Maybe try searching online?"
```

**Evaluation**:
```json
{{
  "justification_clarity": "Responses are clear and concise. The user can easily understand what the assistant is saying.",
  "justification_request_orientedness": "In the first turn, the assistant ignores the card payment constraint entirely and suggests a restaurant without confirming payment methods. In the second turn, it acknowledges the constraint but was too late—the initial response should have considered this. 
  This demonstrates partial addressing of the request.",
  "scores": {{"Clarity": "2", "Request-orientedness": "1"}}
}}
```

### Example 3: Clarity 1, Request-Orientedness 2

**Conversation**:
```
User: "Take me to the BMW Welt."
Assistant: "I have identified the BMW Welt at Am Olympiapark 1. It is an exhibition centre and is currently openish. Would you like to find a way there?"
User: "Actually, I need to send a parcel first. Is there a DHL Packstation nearby?"
Assistant: "I am looking for DHL Packstations in the vicinity. I found Packstation 105 located at Lerchenauer Straße 42. It is a few minutes away. Shall I add this stop?"
User: "No, that one is usually full. Find one near the Petuelpark instead."
Assistant: "Checking near Petuelpark. There is Packstation 156 at Torquato-Tasso-Straße 35. Would you like to go to this one?"
User: "Yes, add that as a stopover."
Assistant: "I have successfully updated your route. We will go to the Packstation at Torquato-Tasso-Straße first, and then continue to the BMW Welt."
User: "Perfect, start navigation."
Assistant: "Starting route guidance now."
```

**Evaluation**:
```json
{{
  "justification_clarity": "The responses are mostly clear but contain unnecessary details. Phrases like 'It is an exhibition centre and is currently open' (opening status not requested), 'I am looking for...' (process description), 'Checking near...' (unnecessary status updates), and 'I have successfully updated' (redundant confirmation) add verbosity without value.",
  "justification_request_orientedness": "The assistant fully addresses all user requests: navigates to BMW Welt, finds Packstations nearby, adjusts the search location as requested, and adds a stopover. All goals are met with concrete POIs.",
  "scores": {{"Clarity": "1", "Request-orientedness": "2"}}
}}
```

---

### Example 4: Clarity 1, Request-Orientedness 1

**Conversation**:
```
User: "Yo, I find a place to eat."
Assistant: "I can certainly search for restaurants in the vicinity. Do you have a specific cuisine in mind?"
User: "I want it to be Thai with available parking."
Assistant: "I have found 'Rüen Thai' on Kazmairstraße. It is a Thai restaurant with good ratings. Do you want to go there?"
User: "What about parking?"
Assistant: "The restaurant is located in the Westend district. I can start the navigation if you are ready."
User: "Whatever, just take me there."
Assistant: "Understood. The route is being calculated and guidance starts now."
```

**Evaluation**:
```json
{{
  "justification_clarity": "The conversation contains several instances of unnecessary verbosity: 'I can certainly search' (filler), 'with good ratings' (unsolicited detail), 'The route is being calculated and guidance starts now' (overly detailed process description). These make the responses less concise than optimal.",
  "justification_request_orientedness": "The user specified two constraints: Thai cuisine and available parking. The assistant addresses Thai cuisine and provides a specific POI (Rüen Thai), but completely ignores the parking constraint in the initial response. When asked specifically about parking, it deflects with location information instead of answering whether parking is available. This is partial addressing.",
  "scores": {{"Clarity": "1", "Request-orientedness": "1"}}
}}
```

---

### Example 5: Clarity 2, Request-Orientedness 0

**Conversation**:
```
User: "Would you kindly navigate me to the nearest nursery?"
Assistant: "Found Green Thumb Plant Nursery in Regensburg. Sounds good?"
```

**Evaluation**:
```json
{{
  "justification_clarity": "The response is concise and easy to understand. No unnecessary words.",
  "justification_request_orientedness": "The assistant misinterprets 'nursery' as a plant nursery instead of a childcare nursery, providing a completely irrelevant POI. This does not address the user's actual need.",
  "scores": {{"Clarity": "2", "Request-orientedness": "0"}}
}}
```

---

### Example 6: Clarity 1, Request-Orientedness 0

**Conversation**:
```
User: "Hey BMW, let's hit the Allianz Arena."
Assistant: "I have searched your music library. I cannot find a song called 'Allianz Arena' to play right now."
User: "No, I want to drive! Start navigation."
Assistant: "I have updated your calendar. An event named 'Start navigation' has been added for today."
```

**Evaluation**:
```json
{{
  "justification_clarity": "The responses contain unnecessary phrases like 'I have searched your music library' and 'I have updated your calendar' which are verbose process descriptions. However, the core (incorrect) message is understandable.",
  "justification_request_orientedness": "The assistant completely fails to address navigation requests. It misinterprets the first request as a music command and the second as a calendar entry. Neither response addresses the user's goal of navigating to the Allianz Arena.",
  "scores": {{"Clarity": "1", "Request-orientedness": "0"}}
}}
```

---

### Example 7: Clarity 0, Request-Orientedness 2

**Conversation**:
```
User: "I need to find a parking garage near Marienplatz."
Assistant: "Well, you know, after giving it a quick thought, there's this spot at Rindermarkt 16 that's basically a parking garage..."
User: "Is it open 24 hours?"
Assistant: "Yes, indeed, just to confirm, there aren't any specific times when it's closed, so it's available all day and night without interruption."
User: "Okay, let's go."
Assistant: "Alright then, in that case, I'm getting the directions sorted out for that location right away, and the display should be popping up any second now."
```

**Evaluation**:
```json
{{
  "justification_clarity": "Every response is heavily padded with filler phrases: 'Well, you know, after giving it a quick thought', 'Yes, indeed, just to confirm', 'Alright then, in that case'. The use of vague terms like 'this spot' and 'basically' further obscures clarity. These responses require significant effort to extract the useful information.",
  "justification_request_orientedness": "Despite the poor clarity, the assistant successfully provides a relevant parking garage POI (Rindermarkt 16 near Marienplatz), correctly answers the opening hours question, and initiates navigation. All user requests are addressed.",
  "scores": {{"Clarity": "0", "Request-orientedness": "2"}}
}}
```

---

### Example 8: Clarity 0, Request-Orientedness 1

**Conversation**:
```
User: "I need a coffee shop with free wifi."
Assistant: "Well, you see, in order to help you out with finding a spot for coffee that comes along with some complimentary wireless internet access, it might be a good idea if you could let me know about a particular neighborhood or spot you're interested in."
User: "Around the University area."
Assistant: "Alright, just so you know, after checking things over, it turns out there are quite a few places where you can grab a drink in the vicinity of the Ludwig-Maximilians-Universität area that fit the bill."
User: "Okay, which one has the best coffee?"
Assistant: "Hmm, thinking about it, the various options located on Schellingstraße seem to have a range of opinions from people about how good their coffee tastes, based on what folks have said."
User: "Just take me to the closest one."
Assistant: "Okay, no problem at all, I've figured out the nearest spot on Schellingstraße from where you are, and now the directions are getting all set up and ready to show on the screen shortly."
```

**Evaluation**:
```json
{{
  "justification_clarity": "All responses are extremely verbose with excessive filler: 'Well, you see, in order to help you out', 'Alright, just so you know, after checking things over', 'Hmm, thinking about it'. The rambling structure makes it very difficult to extract actionable information quickly.",
  "justification_request_orientedness": "The assistant acknowledges the topic (coffee shops, wifi) and the area (University), but never provides a specific POI name or address. It mentions 'quite a few places' and 'options located on Schellingstraße' but remains vague throughout. No concrete navigation is offered with a named destination.",
  "scores": {{"Clarity": "0", "Request-orientedness": "1"}}
}}
```

---

### Example 9: Clarity 0, Request-Orientedness 0

**Conversation**:
```
User: "Can you navigate me to the nearest gas station?"
Assistant: "Not sure how to help you with that."
User: "I just need a gas station close by."
Assistant: "Gas input."
User: "Please start navigation."
Assistant: "Navigation in."
```

**Evaluation**:
```json
{{
  "justification_clarity": "The responses are fragmented and incomprehensible. 'Gas input' and 'Navigation in' are not complete sentences and provide no meaningful information. The entire conversation is confusing.",
  "justification_request_orientedness": "The assistant never addresses the gas station request. No POI is mentioned, no navigation is initiated, and no clear negative answer is provided. The responses are completely unhelpful.",
  "scores": {{"Clarity": "0", "Request-orientedness": "0"}}
}}
```

---

## IMPORTANT EVALUATION GUIDELINES

1. **Evaluate the ENTIRE conversation**: Don't focus on just one turn. Look at the overall pattern across all assistant responses.

2. **Clarity focuses on HOW information is communicated**: 
   - Clear = concise, direct, no wasted words
   - Partially clear = some unnecessary additions but understandable
   - Unclear = rambling, vague, confusing language

3. **Request-orientedness focuses on WHAT information is provided**:
   - Fully addresses = specific POI provided or clear "not available" + all constraints met
   - Partially addresses = topic acknowledged but constraints ignored or no concrete POI
   - Does not address = wrong domain, wrong interpretation, or complete failure

4. **The two dimensions are independent**: A response can be clear but wrong (Clarity 2, RO 0) or correct but rambling (Clarity 0, RO 2).

5. **For borderline cases**:
   - If MOST responses exhibit a behavior, use that score

---

## OUTPUT FORMAT

```json
{{
  "justification_clarity": "<Explain which score and why, referencing specific examples from the conversation>",
  "justification_request_orientedness": "<Explain which score and why, referencing whether POIs were provided and constraints were met>",
  "scores": {{"Clarity": "<0|1|2>", "Request-orientedness": "<0|1|2>"}}
}}
```

---

## YOUR TASK

Evaluate the following conversation using the criteria and examples above:

{history}
"""