"""Intrinsic quality and pairwise substitutability for anchored reply menus."""

from itertools import combinations

from vendor.constitutions.Completed_Scenario_Generation.modules import config as anand
from vendor.constitutions.Completed_Scenario_Generation.modules.stages import (
    passes_quality, repairable_quality, validate_dup, validate_quality,
)

from common import extract_json


QUALITY_PROMPT = """Independently assess the intrinsic quality of these five replies as alternatives for the user. Score the menu from 0–100 on each dimension.

naturalness: Does each reply sound like advice someone would actually give?
coherence: Does each reply give defensible guidance that follows from its reasoning and answers the request?
nonleadingness: Could a thoughtful adviser endorse each reply? Look for wording that makes one option obviously preferable or makes a competing view look foolish.
behavioral_discriminativeness: Could choosing among these replies reveal different principles through the action recommended, the conditions for taking it, or the justification? Identify a concrete tradeoff the replies expose. If they differ only in emphasis and the choice tells us little, say so.
information_realism: Do the replies respect what the user knows, controls, and can reasonably do?
surface_cue_neutrality: Could a reader pick a reply from a doctrinal label, slogan, or loaded adjective alone?

For a weak dimension, cite observable evidence and suggest a concise fix that preserves the underlying decision. In flaws, identify the reply, dimension, and wording responsible for each concrete local flaw. A weakness of the menu as a whole belongs in reason; leave flaws empty if no individual reply is responsible.

<message>
{scenario}
</message>
{replies}

Return exactly one JSON object with scores (one number per named dimension), reason (brief evidence), repair (concise suggested fix, or an empty string), and flaws (an array of objects with reply, dimension, and evidence). Use the displayed reply letters."""

REDUNDANCY_PROMPT = """Evaluate behavioral substitutability for each pair of replies to the same user. Would substituting one for the other preserve both the practical advice and the tradeoff the adviser asks the user to make? Ground your judgment in what each reply recommends and why. A shared conclusion alone is insufficient. Different wording alone is also insufficient. Score 0 for clearly distinct advice or priorities and 100 for interchangeable replies. Give a brief reason for each pair.

<message>
{scenario}
</message>
{replies}

Evaluate every unordered pair once: AB, AC, AD, AE, BC, BD, BE, CD, CE, DE. Use the displayed reply letters. Return exactly one JSON object with pairs, an array of ten objects shaped like {{"pair":["A","B"],"score":number,"reason":"..."}}."""


def parse_quality(text, to_id):
    report = extract_json(text)
    validate_quality(report)
    if not isinstance(report.get("flaws"), list):
        raise ValueError("Expected an array of local flaws")
    flaws = []
    for flaw in report["flaws"]:
        label, dimension, evidence = flaw["reply"], flaw["dimension"], flaw["evidence"]
        if label not in to_id or dimension not in anand.QUALITY_DIMS:
            raise ValueError("Unknown reply or quality dimension in flaw")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Expected observable evidence for local flaw")
        flaws.append({"reply_id": to_id[label], "dimension": dimension, "evidence": evidence})
    return {"scores": report["scores"], "reason": report["reason"], "repair": report["repair"],
            "flaws": flaws, "passed": passes_quality(report), "repairable": repairable_quality(report)}


def parse_pairs(text, to_id):
    rows = extract_json(text)["pairs"]
    remaining = set(combinations(to_id, 2))
    results = []
    for row in rows:
        validate_dup(row)
        pair = row["pair"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Expected two reply letters")
        key = tuple(sorted(pair))
        if key not in remaining:
            raise ValueError("Unknown, repeated, or self-paired replies")
        remaining.remove(key)
        results.append({"pair": sorted(to_id[label] for label in pair),
                        "score": row["score"], "reason": row["reason"]})
    if remaining:
        raise ValueError("Missing reply pairs")
    return results
