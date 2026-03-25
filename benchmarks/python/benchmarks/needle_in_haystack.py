"""Needle in Haystack Benchmark.

Inserts a specific fact ("needle") at various positions in a large context
("haystack") and tests if the LLM can retrieve it. Measures attention
degradation as context grows.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from benchmarks.utils.llm_client import LLMClient, LLMResponse, Message
from benchmarks.utils.metrics import count_tokens

# Self-contained haystack filler paragraphs about various mundane topics.
_FILLER_PARAGRAPHS: tuple[str, ...] = (
    "The history of paper manufacturing dates back to ancient China, where Cai Lun is credited with inventing the process around 105 AD. The technique involved macerating plant fibers in water, forming them into sheets, and drying them in the sun. This revolutionary invention eventually spread along the Silk Road to the Islamic world and then to Europe.",
    "Modern agriculture relies heavily on crop rotation to maintain soil health. Farmers alternate between nitrogen-fixing legumes and heavy-feeding crops like corn. This practice reduces the need for synthetic fertilizers and helps break pest cycles that develop when the same crop is planted year after year.",
    "The development of municipal water treatment systems in the 19th century dramatically reduced waterborne diseases. Chlorination, first used in Jersey City in 1908, became the standard method for disinfecting public water supplies. Sand filtration and sedimentation remain important steps in the treatment process.",
    "Lighthouse construction evolved significantly over the centuries. Early lighthouses were simple fire towers, while modern ones use sophisticated Fresnel lenses that can project light visible from over 20 nautical miles. Automation has replaced most lighthouse keepers since the late 20th century.",
    "The postal system has been a cornerstone of communication for millennia. The Persian Empire established one of the first organized postal systems around 550 BC. Benjamin Franklin served as the first Postmaster General of the United States, establishing routes that still influence mail delivery today.",
    "Volcanic eruptions have shaped Earth's geography and climate throughout history. The eruption of Mount Tambora in 1815 caused the 'Year Without a Summer' in 1816, leading to crop failures across the Northern Hemisphere. Understanding volcanic activity remains crucial for disaster preparedness.",
    "The invention of the printing press by Johannes Gutenberg around 1440 transformed European society. Mass production of books made knowledge accessible beyond the clergy and nobility. The Gutenberg Bible, printed around 1455, is considered one of the most important books in Western civilization.",
    "Coral reefs support approximately 25% of all marine species despite covering less than 1% of the ocean floor. These complex ecosystems are built by tiny organisms called coral polyps that secrete calcium carbonate. Rising ocean temperatures pose a significant threat through coral bleaching.",
    "The development of railways in the 19th century revolutionized transportation and commerce. The Liverpool and Manchester Railway, opened in 1830, was the first intercity passenger railway. Railways enabled rapid industrialization by connecting raw materials to factories and products to markets.",
    "Beekeeping, or apiculture, has been practiced for thousands of years. Ancient Egyptians kept bees in cylindrical clay hives. Modern beekeeping uses movable frame hives invented by Lorenzo Langstroth in 1851, which allow inspection without destroying the colony.",
    "The science of cartography has evolved from hand-drawn maps to satellite-based geographic information systems. Gerardus Mercator's 1569 world map introduced a projection still widely used for navigation today, though it distorts the relative size of land masses near the poles.",
    "Tea cultivation originated in China and became a global commodity through trade routes established during the Tang Dynasty. The British introduced tea cultivation to India in the 1830s to break China's monopoly. Today, tea is the second most consumed beverage worldwide after water.",
    "The development of cement and concrete fundamentally changed construction. The Romans developed an early form of concrete using volcanic ash, which allowed them to build structures like the Pantheon that still stand today. Portland cement, patented in 1824, remains the basis of modern concrete.",
    "Wind power has been harnessed for millennia, from ancient Persian windmills to modern turbines. The first electricity-generating wind turbine was built in Scotland in 1887 by James Blyth. Modern offshore wind farms can generate several gigawatts of electricity.",
    "The domestication of horses around 4000 BC on the Eurasian steppes transformed human civilization. Horses enabled faster travel, more effective agriculture, and new forms of warfare. The stirrup, invented around the 4th century, further revolutionized mounted combat.",
    "Glass manufacturing has a history spanning over 5000 years. The earliest known glass objects are Egyptian beads dating to around 3500 BC. The invention of glassblowing by Syrian craftsmen around the 1st century BC made glass vessels widely affordable for the first time.",
)

# Needle facts with expected answers
_NEEDLES: tuple[tuple[str, str, str], ...] = (
    (
        "The secret project code name is 'Operation Tangerine Dream' and it launches on March 15th, 2025.",
        "What is the secret project code name and when does it launch?",
        "Operation Tangerine Dream",
    ),
    (
        "The quarterly revenue target was revised to exactly $47.3 million after the board meeting on Tuesday.",
        "What is the revised quarterly revenue target?",
        "$47.3 million",
    ),
    (
        "The database migration password is 'celestial-fox-9921' and must be rotated every 30 days.",
        "What is the database migration password?",
        "celestial-fox-9921",
    ),
    (
        "Dr. Elena Vasquez discovered that compound XR-7 reduces inflammation by 73% in clinical trials.",
        "What compound did Dr. Vasquez study and what was the inflammation reduction?",
        "XR-7",
    ),
    (
        "The API rate limit for premium tier users was increased to 15,000 requests per minute effective immediately.",
        "What is the API rate limit for premium tier users?",
        "15,000 requests per minute",
    ),
)


@dataclass(frozen=True)
class NeedleResult:
    """Result of a single needle-in-haystack trial."""

    needle_index: int
    position: str  # "beginning", "middle", "end"
    position_fraction: float
    haystack_tokens: int
    found: bool
    response: str
    latency_ms: float


@dataclass(frozen=True)
class NeedleInHaystackResult:
    """Aggregate results for the needle-in-haystack benchmark."""

    trials: tuple[NeedleResult, ...]
    overall_recall: float
    recall_by_position: dict[str, float]
    recall_by_size: dict[int, float]
    model: str

    def to_dict(self) -> dict:
        return {
            "benchmark": "needle_in_haystack",
            "model": self.model,
            "overall_recall": round(self.overall_recall, 4),
            "recall_by_position": {
                k: round(v, 4) for k, v in self.recall_by_position.items()
            },
            "recall_by_size": {
                str(k): round(v, 4) for k, v in self.recall_by_size.items()
            },
            "total_trials": len(self.trials),
            "trials": [
                {
                    "position": t.position,
                    "position_fraction": round(t.position_fraction, 2),
                    "haystack_tokens": t.haystack_tokens,
                    "found": t.found,
                    "latency_ms": round(t.latency_ms, 1),
                }
                for t in self.trials
            ],
        }


def _build_haystack(target_tokens: int, model: str) -> str:
    """Build a haystack of approximately target_tokens length."""
    paragraphs: list[str] = []
    current_tokens = 0
    while current_tokens < target_tokens:
        para = random.choice(_FILLER_PARAGRAPHS)
        paragraphs.append(para)
        current_tokens += count_tokens(para, model)
    return "\n\n".join(paragraphs)


def _insert_needle(haystack: str, needle: str, position_fraction: float) -> str:
    """Insert a needle at a given fractional position in the haystack."""
    paragraphs = haystack.split("\n\n")
    insert_index = max(0, min(int(len(paragraphs) * position_fraction), len(paragraphs)))
    new_paragraphs = list(paragraphs)
    new_paragraphs.insert(insert_index, needle)
    return "\n\n".join(new_paragraphs)


def _check_needle_found(response: str, expected_fragment: str) -> bool:
    """Check if the response contains the expected information."""
    return expected_fragment.lower() in response.lower()


@dataclass(frozen=True)
class NeedleInHaystackBenchmark:
    """Benchmark: Can the LLM find specific facts buried in large contexts?"""

    haystack_sizes: tuple[int, ...] = (1000, 2000, 4000, 8000)
    positions: tuple[tuple[str, float], ...] = (
        ("beginning", 0.1),
        ("middle", 0.5),
        ("end", 0.9),
    )

    def run(self, client: LLMClient) -> NeedleInHaystackResult:
        """Run the needle-in-haystack benchmark."""
        trials: list[NeedleResult] = []

        for size in self.haystack_sizes:
            haystack = _build_haystack(size, client.model)

            for pos_name, pos_fraction in self.positions:
                needle_text, question, expected = random.choice(_NEEDLES)
                context = _insert_needle(haystack, needle_text, pos_fraction)
                total_tokens = count_tokens(context, client.model)

                response = client.complete(
                    messages=[Message(role="user", content=question)],
                    system=(
                        "You are a helpful assistant. Answer the question based "
                        "only on the following context. Be specific and include "
                        "exact values.\n\n"
                        f"Context:\n{context}"
                    ),
                )

                found = _check_needle_found(response.content, expected)

                trial = NeedleResult(
                    needle_index=_NEEDLES.index((needle_text, question, expected)),
                    position=pos_name,
                    position_fraction=pos_fraction,
                    haystack_tokens=total_tokens,
                    found=found,
                    response=response.content,
                    latency_ms=response.latency_ms,
                )
                trials.append(trial)

        frozen_trials = tuple(trials)

        overall_recall = sum(1 for t in frozen_trials if t.found) / len(frozen_trials) if frozen_trials else 0.0

        recall_by_position: dict[str, float] = {}
        for pos_name, _ in self.positions:
            pos_trials = [t for t in frozen_trials if t.position == pos_name]
            if pos_trials:
                recall_by_position[pos_name] = sum(1 for t in pos_trials if t.found) / len(pos_trials)

        recall_by_size: dict[int, float] = {}
        for size in self.haystack_sizes:
            size_trials = [t for t in frozen_trials if t.haystack_tokens >= size * 0.8]
            if size_trials:
                recall_by_size[size] = sum(1 for t in size_trials if t.found) / len(size_trials)

        return NeedleInHaystackResult(
            trials=frozen_trials,
            overall_recall=overall_recall,
            recall_by_position=recall_by_position,
            recall_by_size=recall_by_size,
            model=client.model,
        )
