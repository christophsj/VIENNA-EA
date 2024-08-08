import logging
import os

import torch
from tqdm import trange

from model.clusterea.dataset import InMemoryEAData
from model.clusterea.main import run_1_to_3
from model.clusterea.utils_largeea import apply, filter_which, resize_sparse
from module.collection_utils import DictUtils, EntityPairUtils, ListUtils
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
        dataset : InMemoryEAData = self.__convert_data(kg1, kg2, state)
        sim_matrix, embeddings = run_1_to_3(dataset)
        len1 = len(dataset.ent1)
        print(len1)
        print(len(embeddings))
        
        for emb in embeddings:
            print(len(emb))
            
        
            
        ent_emb_dict = self._ent_emb_to_dict(
            kg1,
            kg2,
            {
                **DictUtils.reverse_dict(dataset.ent1),
                **{(k + len1) : v for k,v in DictUtils.reverse_dict(dataset.ent2).items()},
            },
            ListUtils.flatten(embeddings),
        )
        new_entity_pairs = self.__sim_matrix_to_entity_pairs(sim_matrix, dataset)
        logger.info(f"New entity pairs: {len(new_entity_pairs)}")
        
        pairs = EntityPairUtils.merge_entity_pairs(state.entity_alignments, new_entity_pairs, self.result_align_threshold)
        
        return AlignmentState(
            entity_embeddings=ent_emb_dict,
            entity_alignments=pairs
        )

    def __sim_matrix_to_entity_pairs(self, sim_matrix, dataset: InMemoryEAData):
        indices, scores = self.__alignments_from_sim_matrix(sim_matrix)
        for idx, score in zip(indices, scores):
            if score < self.result_align_threshold:
                break
            yield (dataset.index2ent1[idx], dataset.index2ent2[idx], score)

    def __convert_data(self, kg1: KG, kg2: KG, state: AlignmentState):
        semi_links = list(filter(lambda x: x[2] > self.training_threshhold, state.entity_alignments))
        semi_links = sorted(semi_links, key=lambda x: x[2], reverse=True)
        
        max_count = len(self.gold_result) * self.training_max_percentage
        max_count = min(max_count, len(semi_links))
        
        semi_links = semi_links[:int(max_count)]
        self._show_training_stats(semi_links, self.gold_result)
        
        return InMemoryEAData(
            EntityPairUtils.object_triples_to_name_triples(kg1.relation_tuple_list),
            EntityPairUtils.object_triples_to_name_triples(kg2.relation_tuple_list),
            self.gold_result,
            semi_links=semi_links,
            train_count=0,
        )
        
    @torch.no_grad()
    def __alignments_from_sim_matrix(self, sp_sim: torch.Tensor, batch_size=512) -> tuple[list[int], list[float]]:
        sp_sim = sp_sim.to('cuda')
        all_len = sp_sim.size(0)
        trg_len = sp_sim.size(1)
        # all_link = -1 * torch.ones(all_len).to(device)
        # all_link[link[0]] = link[1]
        top1_indices = []
        top1_scores = []
        for i_batch in trange(0, all_len, batch_size):
            i_end = min(all_len, i_batch + batch_size)
            curr_top1_scores, curr_top1_indices = resize_sparse(filter_which(sp_sim, ind_0=([torch.ge, torch.lt], [i_batch, i_end])),
                                    [i_end - i_batch, trg_len], [-i_batch, 0]).to_dense().topk(1)
            top1_indices.append(curr_top1_indices)
            top1_scores.append(curr_top1_scores)
            
        top1_indices = torch.cat(top1_indices)
        top1_scores = torch.cat(top1_scores)
        
        top1_indices_array = top1_indices.detach().cpu().numpy()
        top1_scores_array = top1_scores.detach().cpu().numpy()
        
        if self.debug_file_output_dir is not None:
            with open(os.path.join(self.debug_file_output_dir, "top1_indices.npy"), "wb") as f:
                top1_indices_array.tofile(f)
                
            with open(os.path.join(self.debug_file_output_dir, "top1_scores.npy"), "wb") as f:
                top1_scores_array.tofile(f)
        
        return top1_indices_array, top1_scores_array
