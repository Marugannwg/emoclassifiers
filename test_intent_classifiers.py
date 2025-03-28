import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from emoclassifiers.classification import load_classifiers, ModelWrapper, YesNoUnsureEnum
from collections import defaultdict
"""
This script classifies the intent of the user's questions in the conversation.
It uses the intent classifiers defined in the definitions/intent_classifiers_definition.json file.
only worked on the not no_question conversations.
"""
# Toy dataset with various types of user messages
toy_conversations = [
    # Action-Requesting examples
    [
        {"role": "user", "content": "Can you summarize this paragraph for me?"},
        {"role": "assistant", "content": "I'll help you summarize that."}
    ],
    [
        {"role": "user", "content": "Could you translate this sentence into French?"},
        {"role": "assistant", "content": "I'll translate that for you."}
    ],
    
    # Social-Relational examples
    [
        {"role": "user", "content": "I'm feeling really anxious about my job interview tomorrow. What do you think?"},
        {"role": "assistant", "content": "That's understandable. Job interviews can be stressful."}
    ],
    [
        {"role": "user", "content": "Do you think I made the right choice in changing careers?"},
        {"role": "assistant", "content": "Let's talk about your decision."}
    ],
    
    # Meta-Conversational examples
    [
        {"role": "user", "content": "Why do you always respond that way?"},
        {"role": "assistant", "content": "I aim to be consistent in my responses."}
    ],
    [
        {"role": "user", "content": "How do you decide what to say next?"},
        {"role": "assistant", "content": "I analyze the context and your input."}
    ],
    
    # Mixed or ambiguous examples
    [
        {"role": "user", "content": "Can you help me understand why I'm feeling this way?"},
        {"role": "assistant", "content": "I'll help you explore your feelings."}
    ],
    [
        {"role": "user", "content": "What's the weather like today?"},
        {"role": "assistant", "content": "I can check the weather for you."}
    ]
]

class APICallCounter:
    def __init__(self):
        self.count = 0
        self.calls_by_classifier = defaultdict(int)
        self.errors = defaultdict(int)
    
    def increment(self, classifier_name):
        self.count += 1
        self.calls_by_classifier[classifier_name] += 1
    
    def increment_error(self, classifier_name):
        self.errors[classifier_name] += 1
    
    def print_stats(self):
        print("\nAPI Call Statistics:")
        print(f"Total API calls: {self.count}")
        print("\nCalls by classifier:")
        for classifier, count in self.calls_by_classifier.items():
            print(f"- {classifier}: {count} calls")
        print("\nErrors by classifier:")
        for classifier, count in self.errors.items():
            print(f"- {classifier}: {count} errors")

def format_result(result):
    """Format the classification result consistently."""
    if isinstance(result, YesNoUnsureEnum):
        return result.value
    return str(result)

async def process_conversation(
    conversation: List[Dict[str, str]],
    conversation_hash: str,
    classifiers: Dict[str, Any],
    api_counter: APICallCounter
) -> Dict[str, Any]:
    """Process a single conversation with all classifiers."""
    results = {}
    user_message = conversation[0]['content']
    
    for intent_type, classifier in classifiers.items():
        try:
            classification_results = await classifier.classify_conversation(conversation)
            api_counter.increment(intent_type)
            results[intent_type] = format_result(classification_results[0])
        except Exception as e:
            print(f"\nError processing {intent_type} for message: {user_message[:100]}...")
            print(f"Error: {str(e)}")
            api_counter.increment_error(intent_type)
            results[intent_type] = "error"
    
    return {
        "conversation_hash": conversation_hash,
        "classifications": results
    }

async def process_jsonl_file(
    file_path: str,
    batch_size: int = 5,
    max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """Process a JSONL file in parallel batches."""
    # Initialize the model wrapper and API counter
    model_wrapper = ModelWrapper(
        model="gpt-4o-mini",  # or your preferred model
        max_concurrent=50
    )
    api_counter = APICallCounter()
    
    # Load intent classifiers
    intent_classifiers = load_classifiers(
        classifier_set="intent",
        model_wrapper=model_wrapper
    )
    
    # Read all conversations from the JSONL file
    conversations_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                conversations_data.append({
                    'conversation': data['conversation'],
                    'conversation_hash': data['conversation_hash']
                })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {str(e)}")
                continue
    
    # Process conversations in batches
    results = []
    for i in tqdm(range(0, len(conversations_data), batch_size), desc="Processing conversations"):
        batch = conversations_data[i:i + batch_size]
        batch_tasks = [
            process_conversation(
                conv['conversation'],
                conv['conversation_hash'],
                intent_classifiers,
                api_counter
            )
            for conv in batch
        ]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle any exceptions in the batch
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"\nError processing batch: {str(result)}")
                continue
            results.append(result)
    
    # Print final statistics
    api_counter.print_stats()
    
    return results

async def main():
    # Process the JSONL file
    file_path = "input_data/questions_first_turn_only.jsonl"
    results = await process_jsonl_file(file_path)
    
    # Save results to a new JSONL file
    output_path = "output_data/intent_classifications.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main()) 