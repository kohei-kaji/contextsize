import pandas as pd

def calculate_coref_distances(file_path):
    """
    Parses a CoNLL-U file to extract coreference IDs and calculate distances 
    between mentions.
    """
    token_corefs = []
    
    # Step 1: Parse the file and collect (token_id, coref_id) pairs
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            
            token_id = int(parts[0])
            coref_field = parts[4]  # Column 5 in your file contains the Coref ID
            upos = parts[3]  # Column 4 contains the UPOS tag
            
            if coref_field == '_':
                continue
            
            # Coref IDs can be multiple (e.g., "1|2"), so we split by '|'
            ids = coref_field.split('|')
            for cid in ids:
                token_corefs.append((token_id, cid, upos))
    
    # Step 2: Group contiguous tokens into "Mentions"
    # We sort by Coref ID first, then by Token ID
    token_corefs.sort(key=lambda x: (x[1], x[0]))
    
    mentions_map = {} # Dictionary: { coref_id: [[start, end], [start, end], ...] }
    for tid, cid, upos in token_corefs:
        if cid not in mentions_map:
            mentions_map[cid] = [[tid, tid, [upos]]]
        else:
            last_mention = mentions_map[cid][-1]
            # If current token is exactly 1 index after the last token, it's the same mention
            if tid == last_mention[1] + 1:
                last_mention[1] = tid
                last_mention[2].append(upos)
            else:
                mentions_map[cid].append([tid, tid, [upos]])
    
    # Step 3: Calculate distances between consecutive mentions
    distance_records = []
    for cid, mentions in mentions_map.items():
        # Distances only exist if there are at least 2 mentions for an ID
        for i in range(1, len(mentions)):
            prev_m = mentions[i-1]
            curr_m = mentions[i]
            
            prev_id = i - 1
            while prev_m[2][0] == 'PRON' and len(prev_m[2]) == 1:  # Skip if the previous mention is a pronoun
                #print(f"Skipping pronoun mention for coref_id {cid}: {prev_m}")
                prev_id -= 1
                if prev_id < 0:
                    #print(f"No non-pronoun mentions left for coref_id {cid}. Skipping distance calculation.")
                    break
                prev_m = mentions[prev_id]

            # Gap distance: Tokens between the start of A and end of B
            dist_gap = curr_m[1] - prev_m[0]
            
            
            distance_records.append({
                'coref_id': cid,
                'prev_mention_range': f"{prev_m[0]}-{prev_m[1]}",
                'curr_mention_range': f"{curr_m[0]}-{curr_m[1]}",
                'gap_distance': dist_gap
            })
            
    return pd.DataFrame(distance_records)

if __name__ == "__main__":
    from pathlib import Path

    folder = Path('../gpt2_coref_results')

    mean = []
    for file in folder.glob('*.conllu'):
        df = calculate_coref_distances(file)
        mean.append(df['gap_distance'].mean())
    print(sum(mean)/len(mean))
