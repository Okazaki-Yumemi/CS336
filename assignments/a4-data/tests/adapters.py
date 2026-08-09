from __future__ import annotations

import os
from typing import Any
from cs336_data.html2text import extract_text_from_html_bytes
from cs336_data.languageIdentification import identify_language
from cs336_data.Pii import mask_emails,mask_phone_numbers,mask_ips
from cs336_data.harmful import identify_nsfw, identify_hate
from cs336_data.quality import gopher_quality_filter
from cs336_data.quality_classifier import classify_quality
from cs336_data.exactlinededup import exact_line_dedup

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return identify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return identify_hate(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return exact_line_dedup(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
