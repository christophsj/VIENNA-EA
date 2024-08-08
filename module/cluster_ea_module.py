import logging
import os

from model.clusterea.dataset import InMemoryEAData
from model.clusterea.main import run_1_to_3
from module.collection_utils import DictUtils, EntityPairUtils
from module.module import AlignmentState, Module
from objects.KG import KG

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger()


class ClusterEAModule(Module):

    def __init__(
        self,
        dataset_name,
        gold_result,
        training_threshold: float = 0.8,
        training_max_percentage: float = 0.5,
        result_align_threshold=float("-inf"),
        model_path=None,
        debug_file_output_dir=None,
    ):
        self.training_threshhold = training_threshold
        self.training_max_percentage = training_max_percentage
        self.model_path = model_path
        self.result_align_threshold = result_align_threshold
        self.debug_file_output_dir = debug_file_output_dir
        self.dataset_name = dataset_name
        self.gold_result = gold_result

        if debug_file_output_dir is not None:
            os.makedirs(debug_file_output_dir, exist_ok=True)

        logger.info("BertIntModule parameters:")
        logger.info(f"dataset_name: {dataset_name}")
        logger.info(f"training_threshold: {training_threshold}")
        logger.info(f"training_max_percentage: {training_max_percentage}")
        logger.info(f"model_path: {model_path}")
        logger.info(f"result_align_threshold: {result_align_threshold}")
        logger.info(f"debug_file_output: {debug_file_output_dir}")

    def step(self, kg1: KG, kg2: KG, state: AlignmentState) -> AlignmentState:
        ent_emb_dict, entity_pairs = self.run(kg1, kg2, state)

        logger.info(f"New entity pairs: {len(entity_pairs)}")
        new_pairs = EntityPairUtils.merge_entity_pairs(
            state.entity_alignments, entity_pairs
        )
        logger.info(f"Merged entity pairs: {len(new_pairs)}")
        return AlignmentState(
            entity_embeddings=ent_emb_dict, entity_alignments=new_pairs
        )

    def run(self, kg1: KG, kg2: KG, state: AlignmentState):
        dataset = self.convert_data(kg1, kg2)
        sim_matrix, embeddings = run_1_to_3(dataset)
        ent_emb_dict = self._ent_emb_to_dict(
            kg1,
            kg2,
            {
                # **DictUtils.reverse_dict(dataset.ent2index1),
                # **DictUtils.reverse_dict(dataset.ent2index2),
            },
            embeddings,
        )
        entity_pairs = self.__sim_matrix_to_entity_pairs(sim_matrix, dataset)
        return AlignmentState(
            entity_embeddings=ent_emb_dict,
            entity_alignments=EntityPairUtils.merge_entity_pairs(
                state.entity_alignments, entity_pairs
            ),
        )

    def __sim_matrix_to_entity_pairs(self, sim_matrix, dataset):
        entity_pairs_dict = {}
        for i, j in zip(*sim_matrix.nonzero()):
            if (
                sim_matrix[i, j] > self.result_align_threshold
                and i != j
                and entity_pairs_dict.get(
                    (dataset.index2ent1[i], dataset.index2ent2[j]), float("-inf")
                )
                < sim_matrix[i, j]
            ):
                entity_pairs_dict[
                    (
                        dataset.index2ent1[i],
                        dataset.index2ent2[j],
                    )
                ] = sim_matrix[i, j]

        return list(
            map(
                lambda x: (x[0][0], x[0][1], x[1]),
                entity_pairs_dict.items(),
            )
        )

    def convert_data(self, kg1: KG, kg2: KG):
        return InMemoryEAData(
            EntityPairUtils.object_triples_to_name_triples(kg1.relation_tuple_list),
            EntityPairUtils.object_triples_to_name_triples(kg2.relation_tuple_list),
            self.gold_result,
            train_count=self.training_max_percentage,
        )
