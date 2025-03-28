import pandas as pd
import json
from typing import Dict, Any
import ast

def extract_first_user_message(conversation_array):
    """Extract the first user message from the conversation array string."""
    # Convert string representation of array to actual array of dicts
    try:
        # If the input is a string, evaluate it as a literal
        if isinstance(conversation_array, str):
            conversation = ast.literal_eval(conversation_array)
        else:
            conversation = conversation_array
            
        # Find first user message
        for message in conversation:
            if message['role'] == 'user':
                return message['content']
    except:
        return None
    return None

def classify_politeness(df: pd.DataFrame, classifier) -> pd.DataFrame:
    """Add politeness classification columns to the DataFrame."""
    # Create new columns
    df['polite_label'] = None
    df['polite_score'] = None
    
    # Process each row
    for idx in range(len(df)):
        # Extract first user message
        user_message = extract_first_user_message(df.iloc[idx]['conversation'])
        
        if user_message:
            try:
                # Get classification
                output = classifier(user_message)
                if output and len(output) > 0:
                    df.at[idx, 'polite_label'] = output[0]['label']
                    df.at[idx, 'polite_score'] = output[0]['score']
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
                
        # Print progress every 100 rows
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)} rows")
    
    return df

def main():
    # Load the parquet file
    input_parquet = "wildchat_10k_enhanced_with_question_types.parquet"
    output_parquet = "wildchat_10k_enhanced_with_politeness.parquet"
    
    print("Loading parquet file...")
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df)} conversations")
    
    # Import your classifier here
    # from your_module import classifier
    
    print("\nAdding politeness classifications...")
    df = classify_politeness(df, classifier)
    
    # Print some statistics
    print("\nPoliteness Statistics:")
    print("\nLabel Distribution:")
    print(df['polite_label'].value_counts(normalize=True).mul(100).round(1))
    
    print("\nScore Statistics:")
    print(df['polite_score'].describe())
    
    # Save the enhanced DataFrame
    print(f"\nSaving enhanced DataFrame to {output_parquet}")
    df.to_parquet(output_parquet)
    print("Done!")

if __name__ == "__main__":
    main() 