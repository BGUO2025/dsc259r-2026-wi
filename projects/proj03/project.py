# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time
from collections import Counter


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    # HTTP Request --> Response
    res = requests.get(url)

    # Access text content
    text = res.text

    # Hard-coded access
    idx1, idx2 = text.find('***'), text.rfind('***')
    text = text[idx1+3: idx2]
    idx1, idx2 = text.find('***'), text.rfind('***')
    text = text[idx1+3: idx2]

    # Replace per requirement
    text = text.replace('\r\n', '\n')

    # Delay scraping
    time.sleep(0.5)
    return text


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    # Strip the space around
    book_string = book_string.strip()

    # If it's empty string
    if not book_string: return ['\x02', '\x03']

    # Split the string into paragraph by white spaces
    paragraphs = re.split(r'\n\s*\n', book_string)

    # Initialize tokenization
    tokens = []
    # Iterate paragraphs
    for para in paragraphs:
        # Strip the space around again
        para = para.strip()
        # In case it's empty string, ignore
        if not para: continue
        # Start para
        tokens.append('\x02')
        # Capture all alphanumeric characters
        para_tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", para)
        tokens.extend(para_tokens)
        # End para
        tokens.append('\x03')
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):

    def __init__(self, tokens):
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        # Find unique tokens
        unique_tokens = list(set(tokens))
        num_token = len(unique_tokens)
        # Find corresponding uniform probabilities
        unique_probs = [1/num_token] * num_token
        # Feed it to Series
        return pd.Series(
            unique_probs,
            index=unique_tokens
        )

    def probability(self, words):
        # If word not found in corpus, 0 prob
        for word in words:
            if word not in self.mdl.index: return 0
        # uniform prob ^ numbers of words
        return self.mdl.iloc[0] ** len(words)
        
    def sample(self, M):
        num_token = self.mdl.index.shape[0]
        return " ".join(
            self.mdl
            [self.mdl == 1/num_token]
            .sample(M, replace=True)
            .index
            .to_list()
        )


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        from collections import Counter
        # Train by calculate frequency count
        freq_map = Counter(tokens)
        # Type conversion
        token_probs = pd.Series(freq_map.values(), index=freq_map.keys())
        # Return proportion per token
        num_tokens = len(tokens)
        return token_probs / num_tokens
    
    def probability(self, words):
        # Iterate each word
        probs = []
        for word in words:
            # Unseen word has 0 probability
            if word not in self.mdl.index: return 0
            # Store probability
            prob = self.mdl.loc[word]
            probs.append(prob)
        # P(Sentence) = P(A) * P(B) * ... * Multiply Individual
        return np.prod(probs)
        
    def sample(self, M):
        return " ".join(
            self.mdl
            .sample(n=M, replace=True, weights=self.mdl)
            .index
            .to_list()
        )


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N
        self.tokens = tokens 
        
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        ngrams = []
        # Use sliding window to slice N grams from the token
        for i in range(len(tokens) - self.N + 1):
            ngrams.append(tuple(tokens[i: i+self.N]))
        return ngrams
        
    def train(self, ngrams):
        # Count the occurrence of ngram and n1gram
        ngrams_counts = Counter(ngrams)
        n1grams_counts = Counter([ngram[:-1] for ngram in ngrams])

        return (
            # Empty df
            pd.DataFrame({})
            .assign(**{
                # Define ngram
                'ngram': ngrams_counts.keys(),
                # Define n1gram
                'n1gram': lambda df: df['ngram'].apply(lambda ser: ser[:-1]),
                # Define conditional probability for w_n | w_n-1, w_n-2, ..., w_n-(N-1)
                'prob': lambda df: [ngrams_counts[ngram] / n1grams_counts[ngram[:-1]] for ngram in df['ngram'].values],
            })
        )
   
    def probability(self, words):
        ## EDGE CASE: empty sentence get 0 probability
        if len(words) == 0: return 0

        ## Create ngram tokens
        ngram_tokens = self.create_ngrams(words)

        ## Create n1gram tokens
        # Convert to list type for manipulation
        tokens_to_n1gram = list(ngram_tokens[0])
        tokens_to_n1gram.pop()
        # Stop when empty
        while len(tokens_to_n1gram) > 0:
            # Convert to tuple for consistency and then insert to the front
            ngram_tokens.insert(0, tuple(tokens_to_n1gram))
            tokens_to_n1gram.pop()

        ## Compute the ngram probability by taking the product of corresponding conditional probability for ngram tokens
        final_prob = 1
        # Iterate tokens
        for token in ngram_tokens:
            # used for iterative attribute access
            curr = self
            while hasattr(curr, "prev_mdl") and len(token) < curr.N:
                curr = curr.prev_mdl
            # Ngram --> if this is df, specific way to access it
            if isinstance(curr.mdl, pd.DataFrame):
                # Reset index to make it easier
                df = curr.mdl.set_index('ngram')
                # Explicit set index element as tuple to avoid tuple index be mistaken as multi-level indexing
                df.index = df.index.map(tuple)
                # Make it efficient when one probability is already 0
                if token not in df.index: return 0
                final_prob *= df.loc[token, 'prob'].item()       
            # Unigram --> if this is ser, specific way to access it  
            if isinstance(curr.mdl, pd.Series):
                # Do this for consistency to above
                ser = curr.mdl
                if token[0] not in ser.index: return 0
                final_prob *= ser.loc[token[0]].item()
        return final_prob
                
    def sample(self, M):
        # Begin the generated process
        generated = ['\x02']

        # Perform generation for N rounds
        for i in range(M-1):
            # Find context
            split_index = len(generated)-(self.N-1)
            context = tuple(generated[split_index:]) if split_index >= 0 else tuple(generated)

            # Find right LM to match
            curr = self
            while hasattr(curr, 'prev_mdl') and len(context) < curr.N-1:
                curr = curr.prev_mdl

            # Match row in LM
            match_rows = curr.mdl.loc[curr.mdl['n1gram'] == context]

            if match_rows.empty: 
                # If no row found, return empty space
                generated.append('\x03')
            else:
                # Sample the row using conditional probability
                generated.append(
                    match_rows
                    .sample(n=1, weights=curr.mdl['prob'])
                    ['ngram']
                    .item()[-1]
                )
        # End the generated process
        generated.append('\x03')

        return ' '.join(generated)        