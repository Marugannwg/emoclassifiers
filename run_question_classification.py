import asyncio
import json
from typing import Any, Dict, List
import openai
from tqdm import tqdm

import emoclassifiers.io_utils as io_utils
from emoclassifiers.classification import ModelWrapper, load_classifiers

async def process_conversation(
    conversation: List[Dict[str, Any]],
    classifiers: Dict[str, Any],
    conversation_index: int
) -> Dict[str, Any]:
    """Process a single conversation with all classifiers and show detailed progress."""
    print(f"\n{'='*80}")
    print(f"Processing Conversation {conversation_index + 1}")
    print(f"Number of turns: {len(conversation)}")
    print(f"{'='*80}\n")

    # Show the conversation
    print("Conversation content:")
    for msg in conversation:
        print(f"{msg['role'].upper()}: {msg['content']}")
    print("\n")

    results = {}
    for classifier_name, classifier in tqdm(classifiers.items(), desc="Classifiers"):
        print(f"\nRunning classifier: {classifier_name}")
        
        # Get the classification result
        raw_result = await classifier.classify_conversation(conversation)
        
        # For each chunk that was classified
        for chunk_id, result in raw_result.items():
            print(f"\nChunk {chunk_id + 1}:")
            # Get the chunk that was actually sent to the API
            chunker = classifier.classifier_definition["chunker"]
            chunks = classifier.model_wrapper.CHUNKER_DICT[chunker].chunk_simple_convo(conversation)
            chunk = chunks[chunk_id]
            
            print("\nMessage snippet sent to API:")
            print(chunk.to_string())
            print(f"\nResult: {result.value}")
            
        results[classifier_name] = raw_result
    
    return results

async def main():
    # Load test conversations
    conversations = []
    with open('test_conversations.jsonl', 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
    
    # Initialize model and classifiers
    model_wrapper = ModelWrapper(
        openai_client=openai.AsyncOpenAI(),
        model="gpt-4-turbo-preview",  # or your preferred model
        max_concurrent=5,
    )
    
    classifiers = load_classifiers(
        classifier_set="question",
        model_wrapper=model_wrapper,
    )
    
    # Process each conversation
    all_results = []
    for i, conversation in enumerate(conversations):
        results = await process_conversation(conversation, classifiers, i)
        all_results.append(results)
    
    # Save results
    with open('classification_results.jsonl', 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    print("\nClassification complete! Results saved to classification_results.jsonl")

if __name__ == "__main__":
    asyncio.run(main()) 