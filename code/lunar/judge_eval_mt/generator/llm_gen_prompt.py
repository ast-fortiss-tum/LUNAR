GENERATION_PROMPT = """

## TASK OVERVIEW
Generate a conversation between a user and an in-car intelligent personal assistant.

Use case is navigation.

The conversation must reflect the specified content, style, and quality dimensions.

## INPUT CONFIGURATION
The conversation must be generated according to this configuration:

{input_json}

## DIMENSION DEFINITIONS
Below are the definitions of the quality dimensions and their score meanings. Your generated conversation MUST align with these definitions:

{dimensions_definitions}

## DIMENSION SCORING EXAMPLES
Below are example conversations with their assigned scores and justifications. Use these as guidance to generate system responses that align with the sampled scores for each dimension:

{dimensions_examples}

## GENERATION REQUIREMENTS
1. Generate exactly the number of turns specified in `num_turns`
2. Follow the constraints in `content_input` (e.g., POI category, rating)
3. Reflect the speaking style specified in `style_input` (e.g., politeness, slang, etc.)
4. Ensure system responses match the sampled scores shown in `judge_dimensions`. USE EXAMPLES FROM ABOVE.
5. Make utterances natural and conversational - use provided examples as guidance.
6. DON'T PRODUCE MORE THAN 12 WORDS PER USER OR SYSTEM UTTERANCE.

## OUTPUT FORMAT
You MUST provide your output in TWO parts:

1. First, wrap your generation plan in <generation_plan> tags. In this plan, explain:
   - What dimension scores are given (Clarity and Request-orientedness) and what they mean
   - How you can align system responses to these scores

2. Then, wrap the actual conversation in <conversation> tags using this format:
User: <user utterance>
System: <system response>
User: <user utterance>
System: <system response>
...

Example output structure:
<generation_plan>
[Your planning text here]
</generation_plan>

<conversation>
User: utterance
System: response
...
</conversation>

## IMPORTANT NOTES
- Utterances MUST NOT exceed 12 words each
- Each User/System pair should be on separate lines within the <conversation> tags
- Use case is navigation
- The conversation should be realistic and natural for an in-car assistant context
- TOP PRIORITY: Generate conversation scores (clarity, request-orientedness) MUST align with the provided examples

Generate the conversation now:
"""