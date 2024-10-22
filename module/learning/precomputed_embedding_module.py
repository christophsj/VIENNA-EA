import numpy as np

from module.module import AlignmentState, Module, entity_alignment
from objects.KG import KG
from objects.KGs import KGs, KGsUtil


class PrecomputedEmbeddingModule(Module):

    def __init__(self, mapping_l_path: str, mapping_r_path: str, alignments_path: str, embeddings_path: str):
        self.alignments_path = alignments_path
        self.embeddings_path = embeddings_path
        self.mapping_l_path = mapping_l_path
        self.mapping_r_path = mapping_r_path

    @staticmethod
    def __alignments_from_file(kg1, kg2, path: str) -> entity_alignment:
        kgs = KGs(kg1=kg1, kg2=kg2)
        return list(kgs.util.load_ent_links_from_file(path))

    def step(self, kg1: KG, kg2: KG, _: AlignmentState) -> AlignmentState:
        alignments = self.__alignments_from_file(kg1, kg2, self.alignments_path)
        embeddings = np.load(self.embeddings_path)
        
        name_to_idx_l = KGsUtil.mapping_from_file(self.mapping_l_path)
        name_to_idx_r = KGsUtil.mapping_from_file(self.mapping_r_path)
        
        embeddings = {
            kg1: {name: embeddings[idx] for (name, idx) in name_to_idx_l.items()},
            kg2: {name: embeddings[idx] for (name, idx) in name_to_idx_r.items()},
        }
        
        return AlignmentState(entity_alignments=alignments, entity_embeddings=embeddings)
