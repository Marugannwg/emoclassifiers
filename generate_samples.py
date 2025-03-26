import json
import random
import pandas as pd
from pathlib import Path

def clean_conversation(conversation_array):
    """Clean conversation by keeping only role and content."""
    # Convert numpy array to list if needed
    conversation = conversation_array.tolist() if hasattr(conversation_array, 'tolist') else conversation_array
    
    # Extract only role and content from each message
    cleaned_conversation = [
        {
            'role': msg['role'],
            'content': msg['content']
        }
        for msg in conversation
    ]
    return cleaned_conversation

def extract_two_turns(conversation):
    """Extract first two user turns from a conversation."""
    user_turns = []
    all_messages = []
    
    for msg in conversation:
        all_messages.append(msg)
        if msg['role'] == 'user':
            user_turns.append(msg)
        if len(user_turns) >= 2:  # Stop after getting 2 user turns
            break
    
    # Find the index of the second user turn or the end of conversation
    end_idx = len(all_messages)
    user_count = 0
    for i, msg in enumerate(all_messages):
        if msg['role'] == 'user':
            user_count += 1
            if user_count == 2:
                end_idx = i + 1
                break
    
    return all_messages[:end_idx]

def main():
    # Input and output files
    input_file = 'wildchat_10k_enhanced_with_emotions.parquet'
    output_file = 'test_conversations.jsonl'
    
    # Load full dataset
    print("Loading full dataset...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} conversations")
    
    # Clean conversations first
    print("Cleaning conversations...")
    df['cleaned_conversation'] = df['conversation'].apply(clean_conversation)
    
    # Randomly sample 100 conversations
    sample_size = min(100, len(df))
    sampled_indices = random.sample(range(len(df)), sample_size)
    sampled_df = df.iloc[sampled_indices]
    print(f"Randomly sampled {sample_size} conversations")
    
    # Extract first two turns and save
    print(f"Extracting two turns and saving...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in sampled_df.iterrows():
            two_turns = extract_two_turns(row['cleaned_conversation'])
            # Only save if we have at least one user turn
            if any(msg['role'] == 'user' for msg in two_turns):
                output_data = {
                    'conversation_hash': row['conversation_hash'],
                    'conversation': two_turns
                }
                f.write(json.dumps(output_data) + '\n')
    
    print(f"Successfully saved cleaned conversations to {output_file}")

if __name__ == "__main__":
    main() 