import re
import json

# ─────────────────────────────────────────────────────────────
#  Cornell Movie-Dialogs Corpus Preprocessor
#
#  Domain: Movie/Entertainment conversations
#  Source: https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
#
#  Place these 2 files in this folder:
#    - movie_lines.txt
#    - movie_conversations.txt
# ─────────────────────────────────────────────────────────────

SEPARATOR = " +++$+++ "


def clean_text(text):
    """Normalize text: lowercase, strip special chars, collapse whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Remove non-ASCII
    text = text.encode('ascii', errors='ignore').decode('ascii')
    # Keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?\'\-]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_lines(filepath="movie_lines.txt"):
    """
    Parse movie_lines.txt into a dict: lineID → text.

    Each line has format:
        L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
        lineID +++$+++ characterID +++$+++ movieID +++$+++ character_name +++$+++ text
    """
    lines = {}
    errors = 0

    with open(filepath, 'r', encoding='iso-8859-1') as f:
        for raw_line in f:
            parts = raw_line.strip().split(SEPARATOR)
            if len(parts) >= 5:
                line_id = parts[0].strip()
                text = parts[4].strip()
                lines[line_id] = text
            else:
                errors += 1

    print(f"Loaded {len(lines)} lines from {filepath} ({errors} parse errors)")
    return lines


def load_conversations(filepath="movie_conversations.txt"):
    """
    Parse movie_conversations.txt into list of conversations.

    Each line has format:
        u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
        characterID1 +++$+++ characterID2 +++$+++ movieID +++$+++ [list_of_lineIDs]
    """
    conversations = []
    errors = 0

    with open(filepath, 'r', encoding='iso-8859-1') as f:
        for raw_line in f:
            parts = raw_line.strip().split(SEPARATOR)
            if len(parts) >= 4:
                # Parse the list of line IDs: "['L194', 'L195', ...]"
                line_ids_str = parts[3].strip()
                try:
                    line_ids = eval(line_ids_str)  # Safe here — it's always a list of strings
                    conversations.append(line_ids)
                except:
                    errors += 1
            else:
                errors += 1

    print(f"Loaded {len(conversations)} conversations from {filepath} ({errors} parse errors)")
    return conversations


def is_valid_pair(q, a, min_words=3, max_words=40):
    """Filter out too-short, too-long, or empty pairs."""
    if not q or not a:
        return False
    q_len = len(q.split())
    a_len = len(a.split())
    if q_len < min_words or a_len < min_words:
        return False
    if q_len > max_words or a_len > max_words:
        return False
    return True


def process_cornell(lines_file="movie_lines.txt", convos_file="movie_conversations.txt",
                    output_path="processed_data.json"):
    """
    Main pipeline:
    1. Load all movie lines
    2. Load conversation structures
    3. Create consecutive Q/A pairs
    4. Clean, filter, deduplicate
    5. Save as JSON
    """

    # Step 1: Load lines
    lines = load_lines(lines_file)

    # Step 2: Load conversations
    conversations = load_conversations(convos_file)

    # Step 3: Create pairs from consecutive turns
    pairs = []
    for convo_line_ids in conversations:
        for i in range(len(convo_line_ids) - 1):
            q_id = convo_line_ids[i]
            a_id = convo_line_ids[i + 1]

            if q_id not in lines or a_id not in lines:
                continue

            q = clean_text(lines[q_id])
            a = clean_text(lines[a_id])

            if is_valid_pair(q, a):
                pairs.append(f"user: {q} bot: {a}")

    print(f"\nRaw pairs created: {len(pairs)}")

    # Step 4: Deduplicate
    seen = set()
    unique_pairs = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique_pairs.append(p)

    print(f"After deduplication: {len(unique_pairs)}")

    # Step 5: Shuffle
    import random
    random.seed(42)
    random.shuffle(unique_pairs)

    # Stats
    lens = [len(p.split()) for p in unique_pairs]
    print(f"\n{'='*50}")
    print(f"Total unique pairs: {len(unique_pairs)}")
    print(f"Word count — Avg: {sum(lens)/len(lens):.1f}, "
          f"Min: {min(lens)}, Max: {max(lens)}, "
          f"Median: {sorted(lens)[len(lens)//2]}")

    # Show samples
    print(f"\nSamples:")
    for i in [0, 1, 2, len(unique_pairs)//2, -1]:
        print(f"  {unique_pairs[i][:120]}")

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_pairs, f, indent=2)

    print(f"\nSaved to {output_path}")
    return unique_pairs


if __name__ == "__main__":
    process_cornell()