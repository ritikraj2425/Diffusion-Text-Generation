import pandas as pd
import re
import json

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s.,!?\']', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# NOTE: Update the column names below to match your specific CSV from Kaggle!
def process_chat_data(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    # Assuming columns are 'Question' and 'Answer' - CHANGE THESE IF DIFFERENT
    q_col = 'question' 
    a_col = 'answer'
    
    paired_sentences = []
    for _, row in df.iterrows():
        q = clean_text(row[q_col])
        a = clean_text(row[a_col])
        
        # Combine them into a single string so the model learns the transition
        if q and a and len(q.split()) + len(a.split()) < 30:
            formatted_dialogue = f"user: {q} bot: {a}"
            paired_sentences.append(formatted_dialogue)
            
    with open(output_path, 'w') as f:
        json.dump(list(set(paired_sentences)), f, indent=4)
    print(f"Saved {len(paired_sentences)} Q&A pairs!")

if __name__ == "__main__":
    process_chat_data("Conversation.csv", "processed_data.json")