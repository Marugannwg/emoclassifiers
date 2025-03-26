import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any

def load_classification_results(file_path: str) -> Dict[str, Dict[str, str]]:
    """Load classification results and create a dictionary keyed by conversation_hash."""
    results_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            hash_val = result['conversation_hash']
            classifications = result['classifications']
            results_dict[hash_val] = {
                'first_turn_type': classifications.get('0', None),
                'second_turn_type': classifications.get('2', None)
            }
    return results_dict

def merge_classifications(df: pd.DataFrame, classifications: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Merge classification results into the DataFrame using conversation_hash."""
    # Create temporary DataFrames for first and second turn classifications
    classification_df = pd.DataFrame.from_dict(
        classifications,
        orient='index'
    ).reset_index().rename(columns={'index': 'conversation_hash'})
    
    # Merge with original DataFrame
    merged_df = df.merge(
        classification_df,
        on='conversation_hash',
        how='left'
    )
    
    # Fill NaN values for conversations that weren't classified
    merged_df['first_turn_type'] = merged_df['first_turn_type'].fillna('not_classified')
    merged_df['second_turn_type'] = merged_df['second_turn_type'].fillna('not_classified')
    
    return merged_df

def main():
    # Load the original parquet file
    input_parquet = "wildchat_10k_enhanced_with_emotions.parquet"
    classifications_file = "question_classifications.jsonl"
    output_parquet = "wildchat_10k_enhanced_with_question_types.parquet"
    
    print("Loading parquet file...")
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df)} conversations")
    
    print("\nLoading classification results...")
    classifications = load_classification_results(classifications_file)
    print(f"Loaded {len(classifications)} classification results")
    
    print("\nMerging classifications...")
    merged_df = merge_classifications(df, classifications)
    
    # Print some statistics
    total_convs = len(merged_df)
    classified_first = (merged_df['first_turn_type'] != 'not_classified').sum()
    classified_second = (merged_df['second_turn_type'] != 'not_classified').sum()
    
    print("\nClassification Statistics:")
    print(f"Total conversations: {total_convs}")
    print(f"First turn classified: {classified_first} ({classified_first/total_convs*100:.1f}%)")
    print(f"Second turn classified: {classified_second} ({classified_second/total_convs*100:.1f}%)")
    
    # Distribution of question types
    print("\nFirst Turn Type Distribution:")
    print(merged_df['first_turn_type'].value_counts(normalize=True).mul(100).round(1))
    
    print("\nSecond Turn Type Distribution:")
    print(merged_df['second_turn_type'].value_counts(normalize=True).mul(100).round(1))
    
    # Save the merged DataFrame
    print(f"\nSaving merged DataFrame to {output_parquet}")
    merged_df.to_parquet(output_parquet)
    print("Done!")

if __name__ == "__main__":
    main() 