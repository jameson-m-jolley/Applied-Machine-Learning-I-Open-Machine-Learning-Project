import json
import sys
#import nltk
#import regex
import re


def clean_and_split_sent(raw_txt):
    regex = r'[.!?](?:\s+|$)|(?:\n\s*\n)'
    sentences = re.split(regex, raw_txt)
    split_sum = list(map(lambda x: sum(map(len,x.split())),sentences))
    return split_sum


bigram_model = {}

def make_bigram_model(text):
    tokens = clean_and_split_sent(text)

    for i in range(len(tokens) - 1):
        current_word = tokens[i]
        next_word = tokens[i+1]

        # 1. Initialize the word if it's the first time we see it
        if current_word not in bigram_model:
            bigram_model[current_word] = {"\x00": 0}
        
        # 2. Initialize the following word count if new
        if next_word not in bigram_model[current_word]:
            bigram_model[current_word][next_word] = 0
            
        # 3. Increment both
        bigram_model[current_word][next_word] += 1
        bigram_model[current_word]["\x00"] += 1 # this is to not use the name space of Total
   

def main():
    cmdin = sys.stdin.read()
    make_bigram_model(cmdin)
    
    print(json.dumps(bigram_model,))


if __name__ == "__main__":
    main()