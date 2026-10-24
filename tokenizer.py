# tokenizer.py
class BPETokenizer:
    def __init__(self):
        self.merges = {} # Stores our learned rules: e.g., ('e', 'r') -> 'er'
        self.vocab = {}  # Maps tokens to integer IDs: e.g., 'er' -> 257
        
    def get_stats(self, ids):
        # Count frequencies of adjacent pairs
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, new_id):
        # Replace all occurrences of the target pair with the new ID
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(new_id)
                i += 2 # Skip the next token since we merged it
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text, vocab_size):
        # 1. Start with raw bytes (UTF-8 encoding)
        tokens = list(text.encode("utf-8"))
        num_merges = vocab_size - 256 # 256 is the base size for all raw bytes
        
        # 2. Iteratively find and merge the most frequent pairs
        for i in range(num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break
            # Find the pair with the highest count
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            # Save the rule and apply it to our training data
            self.merges[best_pair] = new_id
            tokens = self.merge(tokens, best_pair, new_id)
            
        print(f"Trained {len(self.merges)} merges.")

    def encode(self, text):
        # 1. Start with raw bytes
        tokens = list(text.encode("utf-8"))
        
        # 2. Apply our learned merges in order
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            # Find the pair in our text that was merged earliest in training
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # Nothing else to merge!
                
            new_id = self.merges[pair]
            tokens = self.merge(tokens, pair, new_id)
            
        return tokens