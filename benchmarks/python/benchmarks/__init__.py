from benchmarks.benchmarks.needle_in_haystack import NeedleInHaystackBenchmark
from benchmarks.benchmarks.instruction_adherence import InstructionAdherenceBenchmark
from benchmarks.benchmarks.compression_fidelity import CompressionFidelityBenchmark
from benchmarks.benchmarks.retrieval_relevance import RetrievalRelevanceBenchmark
from benchmarks.benchmarks.token_efficiency import TokenEfficiencyBenchmark

ALL_BENCHMARKS: dict[str, type] = {
    "needle_in_haystack": NeedleInHaystackBenchmark,
    "instruction_adherence": InstructionAdherenceBenchmark,
    "compression_fidelity": CompressionFidelityBenchmark,
    "retrieval_relevance": RetrievalRelevanceBenchmark,
    "token_efficiency": TokenEfficiencyBenchmark,
}

__all__ = [
    "ALL_BENCHMARKS",
    "NeedleInHaystackBenchmark",
    "InstructionAdherenceBenchmark",
    "CompressionFidelityBenchmark",
    "RetrievalRelevanceBenchmark",
    "TokenEfficiencyBenchmark",
]
