from module.alignment.alignment_module import AlignmentModule


from collections.abc import Iterable


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
