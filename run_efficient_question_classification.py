import asyncio
import json
from typing import Any, Dict, List
import openai
from tqdm.auto import tqdm

import emoclassifiers.io_utils as io_utils
from emoclassifiers.classification import ModelWrapper, load_classifiers, QuestionTypeEnum
from emoclassifiers.chunking import CHUNKER_DICT

def convert_enum_to_dict(results: Dict) -> Dict:
    """Convert QuestionTypeEnum values to strings for JSON serialization."""
    converted = {}
    for chunk_id, result in results.items():
        converted[str(chunk_id)] = result.value
    return converted

def prepare_conversation(conversation: List[Dict[str, Any]], max_turns: int = 2) -> List[Dict[str, Any]]:
    """Prepare conversation by keeping only user messages up to max_turns."""
    user_messages = [msg for msg in conversation if msg["role"] == "user"][:max_turns]
    return user_messages

async def process_conversation(
    conversation: List[Dict[str, Any]],
    classifier: Any,
    conversation_index: int
) -> Dict[str, Any]:
    """Process a single conversation with the question tree classifier."""
    print(f"\n{'='*80}")
    print(f"Processing Conversation {conversation_index + 1}")
    print(f"Number of turns: {len(conversation)}")
    print(f"{'='*80}\n")

    # Show the conversation
    print("Conversation content:")
    for msg in conversation:
        print(f"{msg['role'].upper()}: {msg['content']}")
    print("\n")

    # Get the classification result
    raw_result = await classifier.classify_conversation(conversation)
    
    # For each chunk that was classified
    for chunk_id, result in raw_result.items():
        print(f"\nChunk {chunk_id + 1}:")
        # Get the chunk that was actually sent to the API
        chunker = classifier.classifier_definition["chunker"]
        chunks = CHUNKER_DICT[chunker].chunk_simple_convo(conversation)
        chunk = chunks[chunk_id]
        
        print("\nMessage snippet sent to API:")
        print(chunk.to_string())
        print(f"\nQuestion Type: {result.value}")
    
    return raw_result

async def process_conversation_batch(
    conversations: List[Dict[str, Any]],
    classifier: Any,
    batch_start_idx: int,
) -> List[Dict[str, Any]]:
    """Process a batch of conversations with the question tree classifier."""
    batch_results = []
    
    for i, conv_data in enumerate(conversations):
        try:
            # Get the classification result
            raw_result = await classifier.classify_conversation(conv_data['conversation'])
            # Convert enums to strings
            serializable_results = convert_enum_to_dict(raw_result)
            # Add conversation hash to results
            result_with_hash = {
                'conversation_hash': conv_data['conversation_hash'],
                'classifications': serializable_results
            }
            batch_results.append(result_with_hash)
        except Exception as e:
            print(f"\nError processing conversation {batch_start_idx + i + 1}: {str(e)}")
            continue
    
    return batch_results

async def process_all_conversations(conversations: List[Dict[str, Any]], classifier: Any, batch_size: int = 10):
    """Process all conversations in batches with progress bar."""
    all_results = []
    total_api_calls = 0
    
    # Create batches
    batches = [conversations[i:i + batch_size] for i in range(0, len(conversations), batch_size)]
    
    # Process batches with progress bar
    with tqdm(total=len(batches), desc="Processing batches") as pbar:
        for batch_idx, batch in enumerate(batches):
            batch_results = await process_conversation_batch(
                conversations=batch,
                classifier=classifier,
                batch_start_idx=batch_idx * batch_size
            )
            
            all_results.extend(batch_results)
            total_api_calls += sum(len(result['classifications']) for result in batch_results)
            pbar.update(1)
    
    return all_results, total_api_calls

async def main():
    # Load test conversations
    print("Loading conversations...")
    conversations = []
    with open('test_conversations.jsonl', 'r') as f:
        for line in f:
            conv_data = json.loads(line)
            conversations.append(conv_data)
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Initialize model and classifier
    model_wrapper = ModelWrapper(
        openai_client=openai.AsyncOpenAI(),
        model="gpt-4o-mini",
        max_concurrent=50,
    )
    
    classifiers = load_classifiers(
        classifier_set="question_tree",
        model_wrapper=model_wrapper,
    )
    classifier = classifiers["QUESTION_TYPE"]
    
    # Process all conversations in batches
    print("\nStarting classification...")
    all_results, total_api_calls = await process_all_conversations(
        conversations=conversations,
        classifier=classifier,
        batch_size=10
    )
    
    # Save results
    output_file = 'classification_results.jsonl'
    with open(output_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nClassification complete!")
    print(f"Total API calls made: {total_api_calls}")
    print(f"Average API calls per conversation: {total_api_calls/len(conversations):.1f}")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main()) 