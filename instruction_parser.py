from __future__ import annotations

from dataclasses import dataclass
import re


SUPPORTED_PARSERS = {"atr", "lip"}


class InstructionParseError(ValueError):
    pass


@dataclass(frozen=True)
class GarmentTarget:
    key: str
    category: str
    parser_field: str
    supported_parsers: tuple[str, ...]
    synonyms: tuple[str, ...]
    category_negatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class EditPass:
    category: str
    parser_field: str
    parser_type: str
    supported_parsers: tuple[str, ...]
    edit_text: str
    category_negatives: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "parser_field": self.parser_field,
            "parser_type": self.parser_type,
            "supported_parsers": list(self.supported_parsers),
            "edit_text": self.edit_text,
            "category_negatives": list(self.category_negatives),
        }


TARGETS = (
    GarmentTarget(
        key="upper_clothes",
        category="Upper-clothes",
        parser_field="upper_clothes",
        supported_parsers=("atr", "lip"),
        synonyms=(
            "upper clothes",
            "upper clothing",
            "shirt",
            "shirts",
            "top",
            "tops",
            "blouse",
            "blouses",
            "t-shirt",
            "t shirt",
            "tshirt",
            "tee",
            "hoodie",
            "hoodies",
            "sweater",
            "sweaters",
        ),
        category_negatives=("torso changed",),
    ),
    GarmentTarget(
        key="dress",
        category="Dress",
        parser_field="dress",
        supported_parsers=("atr", "lip"),
        synonyms=("dress", "dresses", "gown", "gowns"),
        category_negatives=("legs changed",),
    ),
    GarmentTarget(
        key="skirt",
        category="Skirt",
        parser_field="skirt",
        supported_parsers=("atr", "lip"),
        synonyms=("skirt", "skirts"),
        category_negatives=("legs changed",),
    ),
    GarmentTarget(
        key="pants",
        category="Pants",
        parser_field="pants",
        supported_parsers=("atr", "lip"),
        synonyms=("pants", "trousers", "jeans", "leggings"),
        category_negatives=("legs changed",),
    ),
    GarmentTarget(
        key="coat",
        category="Coat",
        parser_field="coat",
        supported_parsers=("lip",),
        synonyms=("coat", "coats", "jacket", "jackets", "blazer", "blazers", "outerwear"),
        category_negatives=("upper clothes changed",),
    ),
    GarmentTarget(
        key="socks",
        category="Socks",
        parser_field="socks",
        supported_parsers=("lip",),
        synonyms=("sock", "socks", "stocking", "stockings"),
        category_negatives=("shoes changed",),
    ),
    GarmentTarget(
        key="jumpsuits",
        category="Jumpsuits",
        parser_field="jumpsuits",
        supported_parsers=("lip",),
        synonyms=("jumpsuit", "jumpsuits", "romper", "rompers", "onesie", "onesies"),
    ),
)

KEYWORD_TO_TARGET = {
    synonym: target
    for target in TARGETS
    for synonym in target.synonyms
}
KEYWORDS_SORTED = sorted(KEYWORD_TO_TARGET, key=len, reverse=True)
ACTION_WORDS = (
    "change",
    "replace",
    "remove",
    "recolor",
    "re-colour",
    "re-colour",
    "re-color",
    "color",
    "colour",
    "make",
    "turn",
    "transform",
    "edit",
    "modify",
    "swap",
    "restyle",
    "convert",
    "delete",
)
FOLLOWUP_SPLIT_PATTERN = re.compile(
    r"^(?:remove|replace|change|recolor|re-colour|re-color|color|colour|make|turn|transform|edit|modify|swap|restyle|convert|delete)\b",
    re.IGNORECASE,
)


def normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_parser_name(parser_name: str) -> str:
    normalized = parser_name.strip().lower()
    if normalized not in SUPPORTED_PARSERS:
        supported = ", ".join(sorted(SUPPORTED_PARSERS))
        raise InstructionParseError(f"Unsupported parser_model '{parser_name}'. Use one of: {supported}.")
    return normalized


def _count_supported_mentions(text: str) -> int:
    lowered = text.lower()
    return sum(1 for keyword in KEYWORDS_SORTED if re.search(rf"\b{re.escape(keyword)}\b", lowered))


def split_instruction_into_clauses(text: str) -> list[str]:
    normalized = normalize_instruction(text)
    if not normalized:
        return []

    primary_clauses = [
        clause.strip()
        for clause in re.split(r"(?i)\s*(?:,|;|\n|\band then\b|\bthen\b)\s*", normalized)
        if clause.strip()
    ]

    expanded: list[str] = []
    for clause in primary_clauses:
        lowered = clause.lower()
        if " and " not in lowered:
            expanded.append(clause)
            continue

        fragments: list[str] = []
        remaining = clause
        while True:
            match = re.search(r"(?i)\s+\band\b\s+", remaining)
            if not match:
                fragments.append(remaining.strip())
                break

            left = remaining[: match.start()].strip()
            right = remaining[match.end() :].strip()

            if left and right and FOLLOWUP_SPLIT_PATTERN.match(right):
                fragments.append(left)
                remaining = right
                continue

            fragments.append(remaining.strip())
            break

        expanded.extend(fragment for fragment in fragments if fragment)

    return expanded


def _find_mentions(clause: str) -> list[tuple[int, GarmentTarget]]:
    lowered = clause.lower()
    mentions: list[tuple[int, GarmentTarget]] = []
    for keyword in KEYWORDS_SORTED:
        pattern = rf"\b{re.escape(keyword)}\b"
        for match in re.finditer(pattern, lowered):
            mentions.append((match.start(), KEYWORD_TO_TARGET[keyword]))

    deduped: list[tuple[int, GarmentTarget]] = []
    seen_positions: set[tuple[int, str]] = set()
    for start, target in sorted(mentions, key=lambda item: item[0]):
        key = (start, target.key)
        if key not in seen_positions:
            deduped.append((start, target))
            seen_positions.add(key)
    return deduped


def _choose_target_from_mentions(clause: str, mentions: list[tuple[int, GarmentTarget]]) -> GarmentTarget | None:
    if not mentions:
        return None

    lowered = clause.lower()
    if len(mentions) == 1:
        return mentions[0][1]

    for start, target in mentions:
        window_start = max(0, start - 40)
        context = lowered[window_start:start]
        if any(action in context for action in ACTION_WORDS):
            return target

    for start, target in mentions:
        prefix = lowered[max(0, start - 12) : start]
        if not re.search(r"\b(?:to|into|with|as)\s*$", prefix):
            return target

    return mentions[0][1]


def select_parser_for_target(target: GarmentTarget, preferred_parser: str) -> str:
    if preferred_parser in target.supported_parsers:
        return preferred_parser
    return target.supported_parsers[0]


def build_generic_outfit_pass(instruction: str, preferred_parser: str) -> EditPass:
    return EditPass(
        category="Outfit",
        parser_field="upper_clothes",
        parser_type="lip",
        supported_parsers=("lip",),
        edit_text=instruction,
        category_negatives=(),
    )


def parse_instruction(instruction: str, preferred_parser: str = "atr") -> list[EditPass]:
    preferred = validate_parser_name(preferred_parser)
    normalized = normalize_instruction(instruction)
    if not normalized:
        raise InstructionParseError("Instruction cannot be empty.")

    clauses = split_instruction_into_clauses(normalized)
    if not clauses:
        raise InstructionParseError("Instruction cannot be empty.")

    passes: list[EditPass] = []
    saw_supported_mention = _count_supported_mentions(normalized) > 0

    for clause in clauses:
        mentions = _find_mentions(clause)
        target = _choose_target_from_mentions(clause, mentions)

        if target is None:
            if passes:
                previous = passes[-1]
                merged_text = f"{previous.edit_text}, {clause}".strip(", ")
                passes[-1] = EditPass(
                    category=previous.category,
                    parser_field=previous.parser_field,
                    parser_type=previous.parser_type,
                    supported_parsers=previous.supported_parsers,
                    edit_text=merged_text,
                    category_negatives=previous.category_negatives,
                )
                continue
            continue

        parser_type = select_parser_for_target(target, preferred)
        current_pass = EditPass(
            category=target.category,
            parser_field=target.parser_field,
            parser_type=parser_type,
            supported_parsers=target.supported_parsers,
            edit_text=clause.strip(),
            category_negatives=target.category_negatives,
        )

        if passes and passes[-1].parser_field == current_pass.parser_field:
            previous = passes[-1]
            merged_text = f"{previous.edit_text}, {current_pass.edit_text}"
            passes[-1] = EditPass(
                category=previous.category,
                parser_field=previous.parser_field,
                parser_type=previous.parser_type,
                supported_parsers=previous.supported_parsers,
                edit_text=merged_text,
                category_negatives=previous.category_negatives,
            )
        else:
            passes.append(current_pass)

    if not passes:
        if saw_supported_mention:
            raise InstructionParseError("Could not confidently split the clothing edit into one or more passes.")
        return [build_generic_outfit_pass(normalized, preferred)]

    return passes
