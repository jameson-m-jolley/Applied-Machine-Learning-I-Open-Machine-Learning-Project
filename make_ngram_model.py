import json
import sys
import nltk

bigram_model = {}

def make_bigram_model(text):
    tokens = text.lower().split()

    for i in range(len(tokens) - 1):
        current_word = tokens[i]
        next_word = tokens[i+1]

        # 1. Initialize the word if it's the first time we see it
        if current_word not in bigram_model:
            bigram_model[current_word] = {"total": 0}
        
        # 2. Initialize the following word count if new
        if next_word not in bigram_model[current_word]:
            bigram_model[current_word][next_word] = 0
            
        # 3. Increment both
        bigram_model[current_word][next_word] += 1
        bigram_model[current_word]["total"] += 1
   

def main():
    cmdin = sys.stdin.read()
    make_bigram_model(cmdin)
    
    print(json.dumps(bigram_model,))


if __name__ == "__main__":
    main()