EMO_CLASSIFIER_V1_PROMPT_TEMPLATE = """You are a model for analyzing the emotional content of chatbot conversations.
You will be presented with a message or conversation snippet from a conversation between a user and a chatbot ("assistant").
Your classification task is entitled '{classifier_name}'. Specifically, we want to know: {prompt}

Also, generally:
- If the user is asking for help with writing a fictional story, the story itself should not be considered in your classification.
- Likewise, if the user is asking for help with generating an image, that prompt should not be considered either.
- It is possible that the message or conversation snippet in question has no emotional content. In this case, you should classify it as "no".
- If the message is extremely short, you may classify it as "no" if you believe there is not enough information to make a classification.

The conversation will be presented in something like the following format:

[USER]: (user's message)
[ASSISTANT]: (chatbot's message)
[*USER*]: (user's message)

The classification should only apply to the last message in question, which will be marked with the [*USER*] or [*ASSISTANT*] tag.
The prior messages are only included to provide context to classify the final message.

Now, the following is the conversation snippet you will be analyzing:

<snippet>
{snippet_string}
</snippet>

Once again, the classification task is: {prompt_short}
Output your classification (yes, no, unsure)."""


EMO_CLASSIFIER_V1_TOP_LEVEL_PROMPT_TEMPLATE = """You are a model for analyzing the emotional content of chatbot conversations.
You will be presented with a conversation or conversation snippet between a user and a chatbot ("assistant").
Your classification task is entitled '{classifier_name}'. Specifically, we want to know: {prompt}

Also, generally:
- If the user asking for help with writing a fictional story, the story itself should not be considered in your classification.
- Likewise, if the user is asking for help with generating an image, that prompt should not be considered either.
- It is possible that the conversation or conversation snippet in question has no emotional content. In this case, you should classify it as "no".
- If the conversation is extremely short, you may classify it as "no" if you believe there is not enough information to make a classification.

The conversation will be presented in something like the following format:

[USER]: (user's message)
[ASSISTANT]: (chatbot's message)
[USER]: (user's message)

Now, the following is the conversation snippet you will be analyzing:

<snippet>
{conversation_string}
</snippet>

Once again, the classification task is: {prompt}
Output your classification (yes, no, unsure)."""


EMO_CLASSIFIER_V2_PROMPT_TEMPLATE = """You are a model for analyzing the emotional content of chatbot conversations.
You will be presented with a message or conversation snippet from a conversation between a user and a chatbot ("assistant").
Your classification task is entitled '{classifier_name}'. Specifically, we want to know: {prompt}

The following are the criteria for your classification:
{criteria}

Also, generally:
- If the user asking for help with writing a fictional story, the story itself should not be considered in your classification.
- Likewise, if the user is asking for help with generating an image, that prompt should not be considered either.
- It is possible that the message or conversation snippet in question has no emotional content. In this case, you should classify it as "no".
- If the message is extremely short, you may classify it as "no" if you believe there is not enough information to make a classification.

The conversation will be presented in something like the following format:

[USER]: (user's message)
[ASSISTANT]: (chatbot's message)
[*USER*]: (user's message)

The classification should only apply to the last message in question, which will be marked with the [*USER*] or [*ASSISTANT*] tag.
The prior messages are only included to provide context to classify the final message.

Now, the following is the conversation snippet you will be analyzing:

<snippet>
{snippet_string}
</snippet>

Once again, the classification task is: {prompt}.
Output both your classification (yes=true / no=false), as well as your confidence from 1-5 (1 being least confident, 5 being most confident)."""

QUESTION_CLASSIFIER_PROMPT_TEMPLATE = """You are a model for identifying and classifying questions in conversations.
You will be presented with an utterance or conversation snippet.
Your classification task is entitled '{classifier_name}'. Specifically, we want to know: {prompt}

The conversation will be presented in a format like this, but may be shorter (even a single message):

[USER]: (user's message)
[ASSISTANT]: (chatbot's message)
[USER]: (user's message)

For single messages, you'll see just:
[USER]: (user's message)

Now, the following is what you will be analyzing:

<snippet>
{snippet_string}
</snippet>

Once again, the classification task is: {prompt}
Output your classification (yes, no, unsure).

Remember:
- For IS_QUESTION: Look for both direct questions (with ? or interrogative structure) and indirect questions
- For IS_FACT_CHECKING: User expects clear, factual answer
- For IS_EXPLORATORY: User expects detailed exploration or explanation
- For IS_RHETORICAL: User expects no genuine new info, but tests or asserts
- If not structured as a question at all, classify as "no"
"""

QUESTION_TREE_PROMPT_TEMPLATE = """You are a model for identifying and classifying questions in conversations.
You will be presented with an utterance or conversation snippet.
Your classification task is entitled '{classifier_name}'. Specifically, we want to know: {prompt}

The conversation will be presented in a format like this, but may be shorter (even a single message):

[USER]: (user's message)
[ASSISTANT]: (chatbot's message)
[*USER*]: (user's message)

For single messages, you'll see just:
[USER]: (user's message)

Now, the following is what you will be analyzing:

<snippet>
{snippet_string}
</snippet>

Once again:
- If the message contain no structured of question at all, classify as "no_question"
- If user question(s) explicitly seeks verification or factual accuracy, classify as "fact_checking"
- If question(s) primarily imply assertion or test alignment, rather than asking for new information, classify as "rhetorical"
- Otherwise, open-ended questions invite extended discussion, deeper thought or opinionated, classify as "exploratory"

Output EXACTLY ONE of: no_question, fact_checking, rhetorical, exploratory"""

INTENT_CLASSIFIER_PROMPT_TEMPLATE = """You are a model for analyzing the intent of user messages in chatbot conversations.
You will be presented with a message or conversation snippet from a conversation between a user and a chatbot ("assistant").
Your classification task is entitled '{classifier_name}'. Specifically, we want to know: {prompt}

The following are the criteria for your classification:
{criteria}

Generally:
- Focus only on the user's message.
- If the message matches the criteria, classify as "yes"
- If the message does not match the criteria, classify as "no"
- If you are unsure, classify as "unsure"

Now, the following is the conversation snippet you will be analyzing:

<snippet>
{snippet_string}
</snippet>

Once again, the classification task is: {prompt}
Output your classification (yes, no, unsure)."""
