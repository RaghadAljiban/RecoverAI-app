from functools import lru_cache

from telerehab import RehabClassifier


@lru_cache(maxsize=1)
def get_classifier() -> RehabClassifier:
    return RehabClassifier()
