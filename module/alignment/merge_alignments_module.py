from module.alignment.alignment_module import AlignmentModule
from module.collection_utils import EntityPairUtils


from collections.abc import Iterable


class MergeAlignmentsModule(AlignmentModule):

    def merge_entity_pairs(
        entity_pairs: Iterable[tuple[str, str, float]],
        new_entity_pairs: Iterable[tuple[str, str, float]],
        result_align_threshold: float,
    ) -> list[tuple[str, str, float]]:

        scores = {}

        for e1, e2, prob in list(entity_pairs) + list(new_entity_pairs):
            if e1 not in scores:
                scores[e1] = {}

            scores[e1][e2] = scores[e1].get(e2, 0) + prob

        results = []
        for e1, targets in scores.items():
            best_e2 = max(targets, key=targets.get)
            final_prob = min(targets[best_e2], 1.0)

            if final_prob >= result_align_threshold:
                results.append((e1, best_e2, final_prob))

        return results
