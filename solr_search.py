#!/usr/bin/env python3
import re
from typing import Any

import httpx

SOLR_BASE_URL = "http://localhost:8983/solr"

# Filtered down from NLTK stop word list
STOP_WORDS = {"it", "they", "their", "this", "that", "these", "is", "are", "was", "be", "a", "an", "the", "and", "but", "if", "or", "as", "of", "at", "by", "for", "with", "into", "to", "in", "on", "then", "such", "there", "no", "not", "same", "s", "t"}

def split_camel_case(text: str) -> str:
    """
    Split camelCase words into separate words.
    fooBar -> foo Bar
    HTTPSConnection -> HTTPS Connection
    myAPIKey -> my API Key
    """
    # Handle sequences of capitals (like HTTPSConnection -> HTTPS Connection)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    # Handle normal camelCase (like fooBar -> foo Bar)
    text = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', text)
    return text

def remove_stop_words(words: list[str]) -> list[str]:
    """Remove stop words from a list of words"""
    return [word for word in words if word.lower() not in STOP_WORDS]

def escape_solr_special_chars(word: str) -> str:
    special_chars = set('+-&|!(){}[]^"~*?:\\/.-')

    if any(char in special_chars for char in word):
        escaped = word.replace('"', '\\"')
        return f'"{escaped}"'

    return f'{word}'

def query_to_terms(query: str) -> list[str]:
    query = split_camel_case(query)
    return remove_stop_words([escape_solr_special_chars(t) for t in query.lower().split()])


async def run_solr_query(params : dict[str, Any], collection : str):
    solr_url = f"{SOLR_BASE_URL}/{collection}/select"
    async with httpx.AsyncClient() as client:
        response = await client.get(solr_url, params=params)
        response.raise_for_status()
        return response.json()

async def solr_with_termfreq(terms : list[str] | str, collection : str, rows : int =100, *,
                       cond : str = "OR", term_freq : bool=True, params : dict[str, Any]={}):
    if isinstance(terms, str):
        terms = query_to_terms(terms)


    # Build the fl (field list) including original fields plus termfreq for each term
    fl_fields = ['*']
    if term_freq:
        fl_fields += [f'tf_{term}:termfreq(content,"{term}")' for term in terms]

    fl = ",".join(fl_fields)

    query = "content:" + (f" {cond} content:".join(terms))
    req_params = {
        'q': query,
        'fl': fl,
        'rows': rows,
        'sort': 'score desc',
        'wt': 'json'
    }

    req_params.update(params)

    return await run_solr_query(req_params, collection)

# Example:
#
# results = await solr_with_termfreq("one two", "slack")
#
# [solr_search.get_term_freqs(d) for d in results["response"]["docs"])]
#
def get_term_freqs(doc: dict[str, Any]) -> dict[str, int]:
    """
    For an item in solr_results["response"]["docs"], extract term frequencies
    """
    tf_dict : dict[str, int] = {}
    for k, v in doc.items():
        if k.startswith("tf_"):
            term = k[3:]
            tf_dict[term] = v
    return tf_dict

def slack_accept(terms : list[str], tfs: dict[str, int]):
    found  = 0
    for t in terms:
        if t in tfs.keys() and tfs[t] >= 1:
            found += 1

    missing = len(terms) - found

    if missing == 0:
        return found >= 1
    if missing == 1:
        return found >= 2
    if missing == 2:
        return found >= 3
    if missing == 3:
        return found >= 6
    if missing >= 4:
        return False

def filter_like_slack(terms : str | list[str], docs: list[dict[str, Any]]):
    if isinstance(terms, str):
        terms = query_to_terms(terms)
    return [doc for doc in docs
            if slack_accept(terms, get_term_freqs(doc))]
