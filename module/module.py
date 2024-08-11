from abc import abstractmethod
from dataclasses import dataclass, field
import logging
from typing import TypeAlias
import numpy as np

from objects.KG import KG

entity_id: TypeAlias = str
relation_id: TypeAlias = str
confidence: TypeAlias = float
entity_alignment: TypeAlias = list[tuple[entity_id, entity_id, confidence]]
relation_alignment: TypeAlias = list[tuple[relation_id, relation_id, confidence]]
entity_embedding: TypeAlias = dict[KG, dict[entity_id, np.ndarray]]
relation_embedding: TypeAlias = dict[KG, dict[relation_id, np.ndarray]]

logger = logging.getLogger(__name__)


@dataclass
class AlignmentState:
    entity_alignments: entity_alignment = field(default_factory=lambda: [])
    relation_alignments: entity_alignment = field(default_factory=lambda: [])

    entity_embeddings: entity_embedding | None = None
    relation_embeddings: relation_embedding | None = None


class Module:

    @abstractmethod
    def step(self, kg_l: KG, kg_r: KG, state: AlignmentState) -> AlignmentState:
        """Run the module."""
        pass

    # utility functions

    @staticmethod
    def _ent_emb_to_dict(kg1: KG, kg2: KG, index2entity: dict[int, str], ent_emb: list) -> dict[KG, dict[str, list]]:
        logger.info(f"Length of entity embedding: {len(ent_emb)}")

        return {
            KG.get_affiliation(kg1, kg2, index2entity[idx]): {index2entity[idx]: emb}
            for idx, emb in enumerate(ent_emb)
            if idx in index2entity and index2entity[idx] != "<PAD>"
        }

    @staticmethod
    def _show_stats(
        compare_set: list[tuple[str, str]], gold_standard_set: list[tuple[str, str]]
    ) -> None:
        def set_to_dict(s: list[tuple[str, str]]):
            return {x[0]: x[1] for x in s}

        compare_dict = set_to_dict(compare_set)
        gold_dict = set_to_dict(gold_standard_set)

        correct_num = 0
        wrong_num = 0
        unknown = 0
        for k, v in compare_dict.items():
            if k in gold_dict:
                if v == gold_dict[k]:
                    correct_num += 1
                else:
                    wrong_num += 1
            else:
                unknown += 1

        total = correct_num + wrong_num
        precision = correct_num / total
        logger.info(
            f"Data stats: Correct: {correct_num}, Wrong: {wrong_num}, Total: {total}, Precision: {precision}, Unknown: {unknown}"
        )
