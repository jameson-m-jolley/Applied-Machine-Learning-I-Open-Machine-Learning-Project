import sys
import numpy as np
import re

import regex

class args_obj:
    def __init__(self):
        self.txt = ""
        self.print_schema = False
        self.from_pipe = False
        self.label = None
        pass

def parse_args(args):
    index = 1
    ret = args_obj()
    while(index < len(args)):
        if args[index] == "-txt":
            #the next vals is filepath to the txt
            ret.txt = args[index+1]
            index += 1
        elif args[index] == "-schema":
            ret.print_schema = True;
        elif args[index] == "-pipe":
            ret.from_pipe = True
        elif args[index] == "-human":
            ret.label = 'human'
        elif args[index] == "-AI":
            ret.label = "AI"
        else:
            print("useage -")
            print("py [options]")
            print("-txt [filename.txt]")
            print("-schema: prints schema")
            sys.exit()
        index += 1
    return ret

def clean_and_split_sent(raw_txt):
    regex = r'[.!?](?:\s+|$)|(?:\n\s*\n)'
    sentences = re.split(regex, raw_txt)
    split_sum = list(map(lambda x: sum(map(len,x.split())),sentences))
    return split_sum

##nlp functions
def sent_len_avg(raw_txt):
    split = clean_and_split_sent(raw_txt)
    return sum(split)/len(split)

def sent_len_var(raw_txt):
    avg = sent_len_avg(raw_txt)
    clean = clean_and_split_sent(raw_txt)
    np_arr = np.array(clean).astype(float)
    np_arr -= avg
    np_arr = np_arr**2
    return sum(np_arr)/len(np_arr)

def word_count(raw_text):
    return len(raw_text.split())


def word_counts(raw_text):
    tokens = raw_text.split()
    word_counts = {}
    for i in tokens:
        if i.isalpha():
            #this is a char not a word
            pass
        try:
            word_counts[i] +=1
        except:
            word_counts[i] = 1
    return word_counts


def vocab_size(raw_text):
    return len(word_counts(raw_text))

def vocab_density(raw_text):
    return (len(word_counts(raw_text))/word_count(raw_text))

def hapax_ratio(raw_text):
    counts = word_counts(raw_text)
    sum_of_u_char = 0
    total = 0

    for key , val in counts.items():
        #print(f'key;{key} val;{val}')
        if(val == 1):
            sum_of_u_char +=1
        total += 1

    if total <= 0:
        return 0
    
    return sum_of_u_char / total

def word_len_avg(raw_txt):
    tokens = raw_txt.split()
    length = len(tokens);
    return sum(map(len,tokens))/length


def word_len_var(raw_txt):
    words = raw_txt.split()
    if not words:
        return 0.0
    
    # 1. Create an array of the LENGTH of each word
    # This was the missing piece!
    lengths = np.array([len(w) for w in words]).astype(float)
    
    # 2. Use NumPy's built-in variance for speed and safety
    # This handles the (x - avg)**2 / n logic automatically
    return float(np.var(lengths))

def emoji_count(raw_txt):
    return len(regex.findall(r'\p{Extended_Pictographic}', raw_txt))

def ext_pictographic_density(raw_txt):
    return emoji_count(raw_txt)/word_count(raw_txt)

# We join the words with a pipe | (the OR operator)
# and wrap them in \b (word boundaries)
stop_word_pattern = r'\b(' + '|'.join([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", 
    "had", "having", "do", "does", "did", "doing", "can", "could", "will", 
    "would", "shall", "should", "may", "might", "must", "and", "but", "if", 
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", 
    "about", "against", "between", "into", "through", "during", "before", 
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", 
    "off", "over", "under", "again", "further", "then", "once", "a", "an", 
    "the", "any", "all", "both", "each", "few", "more", "most", "other", 
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", 
    "too", "very", "s", "t", "just", "don", "now"
]) + r')\b'

# Pre-compile with IGNORECASE for speed
STOP_RE = re.compile(stop_word_pattern, re.IGNORECASE)

def stopword_count(raw_text):
    return len(STOP_RE.findall(raw_text))

def ttr_score(raw_txt):
#   \b\w+\b will match all complete, individual words within a text
    words = re.findall(r'\b\w+\b', raw_txt.lower())
    if not words:
        return 0.0
    # 2. Calculate Types (unique) and Tokens (total)
    tokens = len(words)
    types = len(set(words))
    
    return types / tokens

def punctuation_count(raw_txt):
    return len(re.findall(r'[~`!@#\$%\^&\*\(\)_\-\+=\{\}\[\]\\\|;:\'",<\.>\/\?]',raw_txt))

def prc_punctuation(regex):
    def return_func(raw_txt):
        total = punctuation_count(raw_txt)
        if total == 0:
            return 0
        return len(re.findall(regex,raw_txt))/total
    return return_func

def main():
    args_ob = parse_args(sys.argv)
    if args_ob.print_schema:
        schema = "sent_len_avg,sent_len_var,word_count,vocab_density,hapax_ratio,ttr_score,word_len_avg,word_len_var,ext_pictographic_density,stopword_count,punc_tilde,punc_backtick,punc_excl,punc_at,punc_hash,punc_dollar,punc_percent,punc_caret,punc_amp,punc_star,punc_lparen,punc_rparen,punc_under,punc_hyphen,punc_plus,punc_equal,punc_lbrace,punc_rbrace,punc_lbracket,punc_rbracket,punc_backslash,punc_pipe,punc_semi,punc_colon,punc_squote,punc_dquote,punc_comma,punc_langle,punc_period,punc_rangle,punc_slash,punc_question,label"
        print(schema)
    if args_ob.from_pipe:
        NLtext = sys.stdin.read()
    else:
        with open(args_ob.txt, 'r',encoding='utf-8') as file:
            NLtext = file.read()

        
    _sent_len_avg = sent_len_avg(NLtext)
    _sent_len_var = sent_len_var(NLtext)
    _word_count = word_count(NLtext)
    _vocab_density = vocab_density(NLtext)
    _hapax_ratio = hapax_ratio(NLtext)
    _ttr_score = ttr_score(NLtext)
    _word_len_avg = word_len_avg(NLtext)
    _word_len_var = word_len_var(NLtext)
    _ext_pictographic_density = ext_pictographic_density(NLtext)
    _stopword_count = stopword_count(NLtext)

    # Your existing list (I cleaned up the backslashes for the Python list format)
    special_chars = [
    '~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '+', 
    '=', '{', '}', '[', ']', '\\', '|', ';', ':', "'", '"', ',', '<', '.', '>', 
    '/', '?'
    ]
    res = {}
    for c in special_chars:
        # Use re.escape so that '*' becomes '\*' for the regex engine
        percentage = prc_punctuation(re.escape(c))(NLtext)
        res[c] = percentage

    row =f"{_sent_len_avg},{_sent_len_var},{_word_count},{_vocab_density},{_hapax_ratio},{_ttr_score},{_word_len_avg},{_word_len_var},{_ext_pictographic_density},{_stopword_count},{res['~']},{res['`']},{res['!']},{res['@']},{res['#']},{res['$']},{res['%']},{res['^']},{res['&']},{res['*']},{res['(']},{res[')']},{res['_']},{res['-']},{res['+']},{res['=']},{res['{']},{res['}']},{res['[']},{res[']']},{res['\\']},{res['|']},{res[';']},{res[':']},{res["'"]},{res['"']},{res[',']},{res['<']},{res['.']},{res['>']},{res['/']},{res['?']},{args_ob.label}"
    print(row)

if __name__ =="__main__":
    main()