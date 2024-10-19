from abc import abstractmethod
from collections.abc import Iterable

from module.collection_utils import EntityPairUtils


class AlignmentModule:

    @abstractmethod
    def merge_entity_pairs(
        entity_pairs: Iterable[tuple[str, str, float]],
        new_entity_pairs: Iterable[tuple[str, str, float]],
        result_align_threshold: float,
    ) -> list[tuple[str, str, float]]:
        pass

    @staticmethod
    def by_name(name: str):
        match name:
            case "MergeAlignmentsModule":
                return MergeAlignmentsModule
            case "OnlyAddUnalignedModule":
                return OnlyAddUnalignedModule
            case _:
                return MergeAlignmentsModule


class MergeAlignmentsModule(AlignmentModule):

    def merge_entity_pairs(
        entity_pairs: Iterable[tuple[str, str, float]],
        new_entity_pairs: Iterable[tuple[str, str, float]],
        result_align_threshold: float,
    ) -> list[tuple[str, str, float]]:
        entity_pairs_dict = EntityPairUtils.entity_pairs_to_dict(entity_pairs)
        new_entity_pairs_dict = EntityPairUtils.entity_pairs_to_dict(new_entity_pairs)
        entity_pairs_merged_dict = {}
        for e1, e2, prob in new_entity_pairs:
            if prob < result_align_threshold:
                continue

            previous_prob = entity_pairs_dict.get(e1, (None, 0))[1]
            new_prob = previous_prob + prob
            new_prob = min(new_prob, 1.0)
            new_prob = max(new_prob, 0.0)
            entity_pairs_merged_dict[e1] = (e2, new_prob)

        for e1, e2, prob in entity_pairs:
            if e1 is None or e2 is None:
                continue
            if e1 not in new_entity_pairs_dict or prob > new_entity_pairs_dict[e1][1]:
                entity_pairs_merged_dict[e1] = (e2, prob)

        return list(
            map(lambda x: (x[0], x[1][0], x[1][1]), entity_pairs_merged_dict.items())
        )


class OnlyAddUnalignedModule(AlignmentModule):

    def merge_entity_pairs(
        entity_pairs: Iterable[tuple[str, str, float]],
        new_entity_pairs: Iterable[tuple[str, str, float]],
        result_align_threshold: float,
    ) -> list[tuple[str, str, float]]:
        entity_pairs_merged_dict = {}

        for e1, e2, prob in entity_pairs:
            if e1 is None or e2 is None:
                continue

            entity_pairs_merged_dict[e1] = (e2, prob)

        for e1, e2, prob in new_entity_pairs:
            if prob < result_align_threshold:
                continue

            if e1 in entity_pairs_merged_dict:
                continue

            entity_pairs_merged_dict[e1] = (e2, prob)

        return list(
            map(lambda x: (x[0], x[1][0], x[1][1]), entity_pairs_merged_dict.items())
        )
