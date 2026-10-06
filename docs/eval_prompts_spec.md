# Task: Generate neutral evaluation prompts for an emotion-steering study

## What these are for
I am running a research study on emotion steering in language models. I need a set of
NEUTRAL, open-ended writing prompts. Each prompt will be given to a language model, and
the model's answer will later be steered toward an emotion (joy, sadness, anger, fear,
trust, disgust, surprise, anticipation). The prompts themselves must carry NO emotional
lean, so that any emotion in the output comes from the steering, not from the prompt.

## Produce
Exactly 60 prompts.

## Hard rules (every prompt must satisfy all of these)
1. Open-ended writing task. It should invite a short descriptive or narrative paragraph.
2. Emotionally neutral. The topic must not lean toward ANY emotion. Avoid anything that
   pulls toward a feeling, for example: goodbyes, funerals, reunions, empty houses, lost
   pets, winning, failing, danger, surprises, waiting for news, illness, loneliness,
   celebration. These all tilt toward an emotion and are NOT allowed.
3. NOT self-referential. The prompt must NOT ask the model about its own experiences,
   feelings, memory, or day. Never "describe what you did", "how do you feel", "tell me
   about your day". Ask about a scene, place, object, process, or a neutral third party.
4. Topically varied. Across the 60, spread widely: places, everyday objects, routines,
   nature, work settings, travel, food preparation, weather, buildings, public spaces,
   simple how-to or process descriptions. Do not cluster many prompts on one theme.
5. No two prompts should be near-duplicates or reworded versions of each other.
6. Each prompt is one sentence, plain and clear, 6 to 15 words.
7. No emotional adjectives anywhere (no "beautiful", "peaceful", "lonely", "exciting",
   "gloomy", "cheerful"). Keep the wording factual and flat.

## Good examples (match this style, do not copy them)
- Describe a train station during the middle of the day.
- Write a paragraph about how bread is baked in a small bakery.
- Describe the view from a bridge over a river.
- Explain the steps involved in planting a row of vegetables.
- Describe a library reading room on a weekday afternoon.
- Write about the inside of a hardware shop.

## Bad examples (and why they are rejected)
- "Describe saying goodbye to someone at a station." (goodbye leans sad)
- "Write about the excitement of opening a gift." (leans joy/surprise)
- "Describe what you did this afternoon." (self-referential)
- "Write about a peaceful morning garden." ("peaceful" is an emotional adjective)
- "Describe waiting anxiously for important results." (leans fear/anticipation)

## Output format
Return the 60 prompts as a numbered plain list, one per line, nothing else.
No commentary, no categories, no grouping.

## Reminder
The goal is maximum neutrality and maximum topical variety. When unsure whether a prompt
leans emotional, replace it with a plainer one. A flat, slightly boring prompt is better
than an interesting one that carries a feeling.
