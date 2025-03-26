import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
import openai
from tqdm.auto import tqdm

import emoclassifiers.io_utils as io_utils
from emoclassifiers.classification import ModelWrapper, load_classifiers, QuestionTypeEnum

def load_conversations(file_path: str, start_idx: int, chunk_size: int) -> List[Dict]:
    """Load a chunk of conversations from JSONL file."""
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip to start position
        for _ in range(start_idx):
            next(f)
        # Read chunk_size lines
        for _ in range(chunk_size):
            try:
                line = next(f)
                conv = json.loads(line.strip())
                conversations.append(conv)
            except StopIteration:
                break
            except json.JSONDecodeError:
                continue
    return conversations

async def process_conversation_batch(
    conversations: List[Dict[str, Any]],
    classifier: Any,
    batch_start_idx: int,
) -> List[Dict[str, Any]]:
    """Process a batch of conversations with the question tree classifier."""
    futures = []
    
    # Create futures for all conversations in the batch
    for i, conv_data in enumerate(conversations):
        futures.append(
            process_single_conversation(
                conv_data=conv_data,
                classifier=classifier,
                conv_idx=batch_start_idx + i
            )
        )
    
    # Process all conversations in parallel
    results = await asyncio.gather(*futures, return_exceptions=True)
    
    # Filter out exceptions and return valid results
    return [r for r in results if not isinstance(r, Exception)]

async def process_single_conversation(
    conv_data: Dict[str, Any],
    classifier: Any,
    conv_idx: int
) -> Dict[str, Any]:
    """Process a single conversation."""
    try:
        # Get the classification result
        raw_result = await classifier.classify_conversation(conv_data['conversation'])
        # Convert enums to strings
        classifications = {str(k): v.value for k, v in raw_result.items()}
        # Add conversation hash to results
        return {
            'conversation_hash': conv_data['conversation_hash'],
            'classifications': classifications
        }
    except Exception as e:
        print(f"\nError processing conversation {conv_idx + 1}: {str(e)}")
        raise

async def process_jsonl_in_chunks(
    input_file: str,
    output_file: str,
    chunk_size: int = 1000,
    batch_size: int = 50,  # Increased batch size since we're properly parallel now
    checkpoint_dir: str = "checkpoints"
):
    """Process large JSONL file in chunks with checkpointing."""
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
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
    
    # Count total lines
    print("Counting total conversations...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    total_chunks = (total_lines + chunk_size - 1) // chunk_size
    
    print(f"Processing {total_lines} conversations in {total_chunks} chunks...")
    
    # Process in chunks
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                
                # Check if checkpoint exists
                checkpoint_file = checkpoint_dir / f"chunk_{chunk_idx}.jsonl"
                if checkpoint_file.exists():
                    # Copy checkpoint content to output file
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            out_f.write(line)
                    pbar.update(1)
                    continue
                
                # Load chunk
                conversations = load_conversations(input_file, start_idx, chunk_size)
                if not conversations:
                    break
                
                # Process in batches
                with open(checkpoint_file, 'w', encoding='utf-8') as checkpoint_f:
                    # Process all batches in the chunk
                    for batch_idx in range(0, len(conversations), batch_size):
                        batch = conversations[batch_idx:batch_idx + batch_size]
                        try:
                            results = await process_conversation_batch(
                                conversations=batch,
                                classifier=classifier,
                                batch_start_idx=start_idx + batch_idx
                            )
                            
                            # Write results to both checkpoint and output
                            for result in results:
                                line = json.dumps(result) + '\n'
                                checkpoint_f.write(line)
                                out_f.write(line)
                                
                        except Exception as e:
                            print(f"\nError processing batch {batch_idx}: {str(e)}")
                            continue
                
                pbar.update(1)
                # Clear memory
                del conversations
    
    print(f"\nProcessing complete! Results saved to {output_file}")

async def main():
    input_file = "input_data/all_conversations_two_turns.jsonl"
    output_file = "question_classifications.jsonl"
    
    print(f"Starting processing of {input_file}")
    print("This will process the file in chunks and save checkpoints for error tolerance")
    
    await process_jsonl_in_chunks(
        input_file=input_file,
        output_file=output_file,
        chunk_size=1000,  # Process 1000 conversations at a time
        batch_size=50,    # Increased: process 50 conversations in parallel
        checkpoint_dir="question_type_checkpoints"
    )

if __name__ == "__main__":
    asyncio.run(main()) 