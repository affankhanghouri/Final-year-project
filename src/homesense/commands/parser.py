"""Conservative full-utterance rules; no LLM, fuzzy action matching, or network."""

import re
from homesense.commands.models import Command, CommandProblem, Intent
from homesense.session import Target

ALIASES = {"phone": Target.PHONE, "mobile": Target.PHONE, "mobile phone": Target.PHONE,
           "cell phone": Target.PHONE, "smartphone": Target.PHONE,
           "key": Target.KEYS, "keys": Target.KEYS,
           "wallet": Target.WALLET, "purse": Target.WALLET}

CANCEL_PHRASES = {"cancel", "cancel search", "stop searching", "stop search"}
REPEAT_PHRASES = {"repeat", "repeat direction", "repeat the direction", "repeat guidance", "say that again"}
HELP_PHRASES = {"help", "help me", "what can you do", "what can you find", "what objects can you find"}
CONFIRM_PHRASES = {"i found it", "i ve got it", "i have got it", "i got it", "i have found it"}
FIND_VERBS = ("find", "locate", "search for", "look for", "where is", "where are")
POLITE_PREFIXES = ("", "please ", "can you ", "can you please ", "could you ", "could you please ")


def words(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower().replace("’", "'")))


def command_words(text: str) -> str:
    """Strip only documented, anchored courtesy words and the optional prefix."""
    text = words(text)
    text = re.sub(r"^(?:hey agent|heyagent)(?: |$)", "", text, count=1)
    text = re.sub(r"^(?:can you|could you) ", "", text, count=1)
    text = re.sub(r"^please | please$", "", text).strip()
    return text


def command_grammar() -> tuple[str, ...]:
    """Production command vocabulary, including unknown and rejection contexts.

    Vosk learns a word graph from these phrases, not an exact phrase allowlist.
    The parser remains the acceptance boundary. Unknown targets use the existing
    unrestricted decoder only for clarification/veto, never to invent a FIND.
    """
    controls = CANCEL_PHRASES | REPEAT_PHRASES | HELP_PHRASES | (CONFIRM_PHRASES - {"i ve got it"})
    requests = {f"{verb} {article}{alias}" for verb in FIND_VERBS
                for article in ("", "my ", "the ") for alias in ALIASES}
    requests.update(f"i found my {alias}" for alias in ALIASES)
    # Give negations, ambiguity and open targets a path rather than forcing them
    # into a supported target. These phrases are deliberately NOT accepted rules.
    rejects = {f"find my {target}" for target in ("[unk]", "clothes", "glasses", "shoes", "bag")}
    rejects.update(f"do not {phrase}" for phrase in ("find my phone", "find my keys", "find my wallet", "cancel", "stop searching"))
    rejects.update(("don't cancel", "don't stop searching", "i have not found it", "never find my phone",
                    "find my phone or keys", "find my phone and wallet"))
    base = {prefix + phrase for phrase in controls | requests | rejects for prefix in POLITE_PREFIXES}
    return tuple(sorted(base | {"hey agent"} | {"hey agent " + phrase for phrase in base})) + ("[unk]",)


def parse_command(text: str) -> Command:
    if "[unk]" in text.lower():
        return Command(Intent.UNKNOWN, problem=CommandProblem.UNRECOGNIZED)
    text = command_words(text)
    if not text:
        return Command(Intent.UNKNOWN, problem=CommandProblem.EMPTY)
    if text in CANCEL_PHRASES:
        return Command(Intent.CANCEL_SEARCH)
    if text in REPEAT_PHRASES:
        return Command(Intent.REPEAT_GUIDANCE)
    if text in HELP_PHRASES:
        return Command(Intent.HELP)
    if text in CONFIRM_PHRASES:
        return Command(Intent.CONFIRM_FOUND)
    confirm = re.fullmatch(r"i (?:have )?found (?:my |the )?(.+)", text)
    find = re.fullmatch(r"(?:find|locate|search for|look for|where is|where are) (?:my |the |a |an )?(.+)", text)
    match = confirm or find
    if match:
        target_text = match.group(1)
        candidates = tuple(t for t in Target if any(
            re.search(r"\b" + re.escape(alias) + r"\b", target_text) for alias, canonical in ALIASES.items() if canonical == t))
        if len(candidates) > 1 or re.search(r"\b(?:and|or)\b", target_text):
            return Command(Intent.UNKNOWN, problem=CommandProblem.AMBIGUOUS, candidates=candidates)
        target = ALIASES.get(target_text)
        if target:
            return Command(Intent.CONFIRM_FOUND if confirm else Intent.FIND_OBJECT, target)
        return Command(Intent.UNKNOWN, problem=CommandProblem.UNSUPPORTED)
    return Command(Intent.UNKNOWN, problem=CommandProblem.UNRECOGNIZED)
