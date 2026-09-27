"""Conservative Arabic text normalization."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_ARABIC_DIACRITICS = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationOptions:
    """Controls potentially meaning-changing normalization steps."""

    normalize_alef: bool = True
    normalize_yeh: bool = True
    normalize_teh_marbuta: bool = False


def normalize_arabic(
    text: str, options: NormalizationOptions | None = None
) -> str:
    """Return a deterministic normalized representation of Arabic text.

    Unicode is first normalized to NFC. Arabic combining marks and tatweel are
    removed, selected spelling variants are mapped, and whitespace is collapsed.
    Punctuation and digits are preserved so evaluation does not hide those errors.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    options = options or NormalizationOptions()
    result = unicodedata.normalize("NFC", text)
    result = _ARABIC_DIACRITICS.sub("", result).replace("ـ", "")

    if options.normalize_alef:
        result = result.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}))
    if options.normalize_yeh:
        result = result.replace("ى", "ي")
    if options.normalize_teh_marbuta:
        result = result.replace("ة", "ه")

    return _WHITESPACE.sub(" ", result).strip()
