import pandas as pd
from pathlib import Path
import json
from tqdm.auto import tqdm

def clean_conversation(conversation_array):
    """Clean conversation by keeping only role and content."""
    conversation = conversation_array.tolist() if hasattr(conversation_array, 'tolist') else conversation_array
    return [{'role': msg['role'], 'content': msg['content']} for msg in conversation]

def extract_two_turns(conversation):
    """Extract first two user turns and all messages up to second user turn."""
    user_turns = []
    all_messages = []
    
    for msg in conversation:
        all_messages.append(msg)
        if msg['role'] == 'user':
            user_turns.append(msg)
        if len(user_turns) >= 2:
            break
    
    # Find index of second user turn or end
    end_idx = len(all_messages)
    user_count = 0
    for i, msg in enumerate(all_messages):
        if msg['role'] == 'user':
            user_count += 1
            if user_count == 2:
                end_idx = i + 1
                break
    
    return all_messages[:end_idx]

def process_parquet_to_jsonl(
    input_file: str,
    output_file: str,
    chunk_size: int = 1000,
):
    """Extract conversations from parquet file to JSONL format."""
    # Create output directory if needed
    output_path = Path(output_file).parent
    output_path.mkdir(exist_ok=True)
    
    print("Loading parquet file...")
    df = pd.read_parquet(input_file)
    total_rows = len(df)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    print(f"Processing {total_rows} conversations in {total_chunks} chunks...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        with tqdm(total=total_rows, desc="Extracting conversations") as pbar:
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_rows)
                
                # Process chunk
                df_chunk = df.iloc[start_idx:end_idx]
                
                for _, row in df_chunk.iterrows():
                    try:
                        # Clean and extract conversations
                        cleaned_conv = clean_conversation(row['conversation'])
                        two_turns = extract_two_turns(cleaned_conv)
                        
                        # Only save if we have at least one user turn
                        if any(msg['role'] == 'user' for msg in two_turns):
                            output_data = {
                                'conversation_hash': row['conversation_hash'],
                                'conversation': two_turns
                            }
                            f.write(json.dumps(output_data) + '\n')
                    except Exception as e:
                        print(f"\nError processing conversation: {str(e)}")
                        continue
                    
                    pbar.update(1)
                
                # Clear memory
                del df_chunk
    
    print(f"\nExtraction complete! Conversations saved to {output_file}")

def main():
    input_file = "wildchat_10k_enhanced_with_emotions.parquet"
    output_file = "all_conversations_two_turns.jsonl"
    
    print(f"Starting extraction from {input_file}")
    process_parquet_to_jsonl(
        input_file=input_file,
        output_file=output_file,
        chunk_size=1000
    )

if __name__ == "__main__":
    main() 