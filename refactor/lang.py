import json
import logging
from typing import Any, cast

import bm25s  # pyright: ignore[reportMissingTypeStubs]
import jieba  # pyright: ignore[reportMissingTypeStubs]
import Stemmer
from bm25s.tokenization import Tokenized  # pyright: ignore[reportMissingTypeStubs]
from lingua import (
    Language,
    LanguageDetectorBuilder,
)


STEMMER_SUPPORTED_LANGUAGES = [
    "zh",  # Custom
    "tr",
    "fr",
    "sv",
    "no",
    "ru",
    "id",
    "nl",
    "ca",
    "ro",
    "en",
    "hi",
    "pt",
    "it",
    "da",
    "hu",
    "ta",
    "es",
    "de",
    "ar",
    "lt",
    "fi",
]

min_confidence = 0.1

logger = None

# use high-accuracy for text <= 120 characters: https://github.com/pemistahl/lingua-py?tab=readme-ov-file#112-minimum-relative-distance
logging.getLogger("jieba").setLevel(logging.WARNING)
jieba.initialize()  # pyright: ignore[reportUnknownMemberType]
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.FRENCH,
    Language.SWEDISH,
    Language.NYNORSK,
    Language.BOKMAL,
    Language.DUTCH,
    Language.DANISH,
    Language.GERMAN,
    Language.SPANISH,
    Language.PORTUGUESE,
    Language.ARABIC,
    Language.RUSSIAN,
    Language.CHINESE,
).build()


def detect_language(input_string: str) -> str:
    if len(input_string) >= 100_000:
        input_string = input_string[:40_000] + " " + input_string[-40_000:]

    detected_language = detector.detect_language_of(input_string)

    if detected_language is None:
        logger.warning(
            f"[detect_language]: Could not identify language. Defaulting to english: {json.dumps(input_string[:200])}"
        )
        return "en"

    probability = detector.compute_language_confidence(input_string, detected_language)

    # Get language code
    language_code = detected_language.iso_code_639_1.name.lower()
    LANGUAGE_CODE_MAP = {
        "nno": "no",
        "nb": "no",
        "nbo": "no",
        "nn": "no",
    }
    language_code = LANGUAGE_CODE_MAP.get(language_code, language_code)

    # default to english if lingua is unsure
    if probability < min_confidence or language_code not in STEMMER_SUPPORTED_LANGUAGES:
        msg = (
            "Could not confidently determine a valid language"
            if probability < min_confidence
            else "Predicted language is not supported by the Stemmer"
        )
        logger.warning(
            f"[detect_language]: {msg}. Best guess is {detected_language.name} ({language_code}), confidence = {probability} on string {json.dumps(input_string[:100])}. Defaulting to english."
        )
        return "en"

    return language_code


def generate_stemmed_string_frequencies(
    input_string: str,
    *,
    language: str,
) -> dict[str, int]:
    token_to_freq: dict[str, int] = {}
    if language == "zh":
        words = cast(list[Any], list(jieba.cut(input_string, HMM=True)))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        for word in words:
            assert isinstance(word, str)
            word = word.strip()
            if len(word) == 0:
                continue
            if word not in token_to_freq:
                token_to_freq[word] = 0
            token_to_freq[word] += 1
    else:
        stemmer = cast(Any, Stemmer.Stemmer(language))  # pyright: ignore[reportUnknownMemberType]
        query_tokenized = bm25s.tokenize(  # pyright: ignore[reportUnknownMemberType]
            input_string,
            stopwords=[],
            stemmer=stemmer,
            show_progress=False,
        )
        assert isinstance(query_tokenized, Tokenized)
        token_id_to_token: dict[int, str] = {}
        for token, token_id in query_tokenized.vocab.items():
            token_id_to_token[token_id] = token
        for token_id in query_tokenized.ids[0]:
            token = token_id_to_token[token_id]
            token_to_freq[token] = token_to_freq.get(token, 0) + 1

    return token_to_freq


def generate_stemmed_string(input_string: str, *, language: str) -> list[str]:
    query_tokenized = generate_stemmed_string_frequencies(
        input_string,
        language=language,
    )
    tokens: list[str] = []
    for token, freq in query_tokenized.items():
        for _ in range(freq):
            tokens.append(token)
    return tokens
