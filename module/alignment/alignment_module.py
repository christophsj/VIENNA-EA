from abc import abstractmethod
from collections.abc import Iterable


class AlignmentModule:

    @abstractmethod
    def merge_entity_pairs(
        entity_pairs: Iterable[tuple[str, str, float]],
        new_entity_pairs: Iterable[tuple[str, str, float]],
        result_align_threshold: float,
    ) -> list[tuple[str, str, float]]:
        pass
