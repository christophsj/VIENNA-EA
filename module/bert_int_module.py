import logging
import os
import re
from typing import Optional

import torch
from transformers import BertTokenizer

from model.bert_int.Basic_Bert_Unit_model import Basic_Bert_Unit_model
from model.bert_int.basic_bert_unit.Read_data_func import (
    ent2desTokens_generate,
    ent2Tokens_gene,
    ent2bert_input,
    ent2desTokens_generateFromDict,
    get_name,
)
from model.bert_int.basic_bert_unit.main import train_basic_bert
from model.bert_int.bert_int_input import BertIntInput
from model.bert_int.interaction_model.Param import (
    BASIC_BERT_UNIT_MODEL_OUTPUT_DIM,
    CUDA_NUM,
)
from model.bert_int.interaction_model.clean_attribute_data import clean_attribute_data
from model.bert_int.interaction_model.get_attributeValue_embedding import (
    get_attribute_value_embedding,
)
from model.bert_int.interaction_model.get_attributeView_interaction_feature import (
    get_attributeView_interaction_feature,
)
from model.bert_int.interaction_model.get_entity_embedding import (
    main as get_entity_embedding_main,
)
from model.bert_int.interaction_model.get_neighView_and_desView_interaction_feature import (
    get_neightview_and_desview_interaction_feature,
)
from model.bert_int.interaction_model.interaction_model import (
    interaction_model as train_interaction_model,
)
from module.collection_utils import DictUtils, EntityPairUtils, ListUtils
from module.module import AlignmentState, Module
from objects.KG import KG

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger()


class BertIntModule(Module):

    def __init__(
        self,
        dataset_name: str,
        des_dict_path: str | None = None,
        description_name_1: str = None,
        description_name_2: str = None,
        training_threshold: float = 0.8,
        training_max_percentage: float = 0.5,
        result_align_threshold=float("-inf"),
        model_path=None,
        interaction_model=True,
        debug_file_output_dir=None,
        gold_result=None,
    ):
        self.des_dict_path = des_dict_path
        self.training_threshhold = training_threshold
        self.training_max_percentage = training_max_percentage
        self.description_name_l = description_name_1
        self.description_name_r = description_name_2
        self.model_path = model_path
        self.interaction_model = interaction_model
        self.result_align_threshold = result_align_threshold
        self.debug_file_output_dir = debug_file_output_dir
        self.dataset_name = dataset_name
        self.gold_result = gold_result

        if debug_file_output_dir is not None:
            os.makedirs(debug_file_output_dir, exist_ok=True)

        logger.info("BertIntModule parameters:")
        logger.info(f"dataset_name: {dataset_name}")
        logger.info(f"des_dict_path: {des_dict_path}")
        logger.info(f"training_threshold: {training_threshold}")
        logger.info(f"training_max_percentage: {training_max_percentage}")
        logger.info(f"description_name_1: {description_name_1}")
        logger.info(f"description_name_2: {description_name_2}")
        logger.info(f"model_path: {model_path}")
        logger.info(f"interaction_model: {interaction_model}")
        logger.info(f"result_align_threshold: {result_align_threshold}")
        logger.info(f"debug_file_output: {debug_file_output_dir}")

    @staticmethod
    def __load_model_from_path(bert_model_path: str) -> Basic_Bert_Unit_model:
        Model = Basic_Bert_Unit_model(768, BASIC_BERT_UNIT_MODEL_OUTPUT_DIM)
        Model.load_state_dict(torch.load(bert_model_path, map_location="cpu"))
        logger.info("loading basic bert unit model from:  {}".format(bert_model_path))
        Model.eval()
        for name, v in Model.named_parameters():
            v.requires_grad = False
        Model = Model.cuda(CUDA_NUM)
        return Model

    def __convert_basic_unit_to_final(
        self, bert_int_data: BertIntInput, entity_pairs: list[tuple[str, str, float]]
    ) -> list[tuple[str, str, float]]:
        entity_pairs_dict = EntityPairUtils.entity_pairs_to_candidate_dict(entity_pairs)

        if self.debug_file_output_dir is not None:
            with open(f"{self.debug_file_output_dir}/candidates.csv", "w") as f:
                for e1, candidate_list in sorted(
                    entity_pairs_dict.items(),
                    key=lambda x: bert_int_data.index2entity[x[0]],
                ):
                    candidate_list_name = list(
                        map(
                            lambda x: f"{bert_int_data.index2entity[x[0]]}:{x[1]}",
                            candidate_list,
                        )
                    )
                    candidate_list_name_joined = "\t".join(candidate_list_name)
                    f.write(
                        f"{bert_int_data.index2entity[e1]}\t{candidate_list_name_joined}\n"
                    )

        entity_pairs = map(
            lambda x: (x[0], x[1][0][0], x[1][0][1]), entity_pairs_dict.items()
        )
        entity_pairs = list(
            map(
                lambda x: (
                    bert_int_data.index2entity[x[0]],
                    bert_int_data.index2entity[x[1]],
                    x[2],
                ),
                entity_pairs,
            )
        )
        return entity_pairs

    def step(self, kg1: KG, kg2: KG, state: AlignmentState) -> AlignmentState:
        if self.interaction_model:
            _, ent_emb_dict, entity_pairs = self.run_interaction_model(kg1, kg2, state)
        else:
            bert_int_data, _, _, entity_pairs, ent_emb_dict, _, _ = self.run_basic_unit(
                kg1, kg2, state
            )
            entity_pairs = self.__convert_basic_unit_to_final(
                bert_int_data, entity_pairs
            )

        logger.info(f"New entity pairs: {len(entity_pairs)}")
        new_pairs = EntityPairUtils.merge_entity_pairs(
            state.entity_alignments, entity_pairs, self.result_align_threshold
        )
        logger.info(f"Merged entity pairs: {len(new_pairs)}")
        return AlignmentState(
            entity_embeddings=ent_emb_dict, entity_alignments=new_pairs
        )

    def run_interaction_model(self, kg1: KG, kg2: KG, state: AlignmentState):
        (
            bert_int_data,
            trained_module,
            ent_emb,
            entity_pairs,
            ent_emb_dict,
            train_candidates,
            test_candidates,
        ) = self.run_basic_unit(kg1, kg2, state)

        keep_attr_1, keep_attr_2, remove_data_1, remove_data_2 = clean_attribute_data(
            kg1.attribute_tuple_list, kg2.attribute_tuple_list
        )
        att_datas = list(
            map(
                lambda x: (bert_int_data.entity2index[x[0]], x[1], x[2], x[3]),
                [*keep_attr_1, *keep_attr_2],
            )
        )
        value_emb, value_set = get_attribute_value_embedding(
            Model=trained_module,
            att_datas=att_datas,
        )
        entity_pairs_without_score = list(map(lambda x: (x[0], x[1]), entity_pairs))
        neighViewInterF, desViewInterF = get_neightview_and_desview_interaction_feature(
            input_data=bert_int_data,
            ent_emb=ent_emb,
            entity_pairs=entity_pairs_without_score,
        )
        attrViewF = get_attributeView_interaction_feature(
            input_data=bert_int_data,
            att_datas=att_datas,
            value_emb=value_emb,
            entity_pairs=entity_pairs_without_score,
            value_list=value_set,
        )
        e1_to_e2_dict = train_interaction_model(
            bert_int_data=bert_int_data,
            entity_pairs=entity_pairs_without_score,
            att_features=attrViewF,
            des_features=desViewInterF,
            nei_features=neighViewInterF,
            test_candidate=test_candidates,
            train_candidate=train_candidates,
        )

        if self.debug_file_output_dir is not None:
            with open(f"{self.debug_file_output_dir}/e1_to_e2_dict.csv", "w") as f:
                for e1, candidate_list in sorted(
                    e1_to_e2_dict.items(),
                    key=lambda x: bert_int_data.index2entity[x[0]],
                ):
                    candidate_list_name = list(
                        map(
                            lambda x: f"{bert_int_data.index2entity[x[0]]}:{x[1]}",
                            candidate_list,
                        )
                    )
                    candidate_list_name_joined = "\t".join(candidate_list_name)
                    f.write(
                        f"{bert_int_data.index2entity[e1]}\t{candidate_list_name_joined}\n"
                    )

        count = 0
        entity_pairs_by_name = []
        for e1, candidate_list in e1_to_e2_dict.items():
            e2, score = EntityPairUtils.take_max_candidate(candidate_list)
            if e2 is not None:
                entity_pairs_by_name.append(
                    (
                        bert_int_data.index2entity[e1],
                        bert_int_data.index2entity[e2],
                        score,
                    )
                )
            else:
                count = count + 1
                logger.warning(f"{bert_int_data.index2entity[e1]} without candidate")

                if len(candidate_list) > 0:
                    logger.warning(f"{candidate_list}")

        logger.warning(f"{count} times without candidate")

        if self.debug_file_output_dir is not None:
            with open(
                f"{self.debug_file_output_dir}/entity_pairs_by_name.csv", "w"
            ) as f:
                for e1, e2, score in sorted(
                    entity_pairs_by_name, key=lambda x: (x[0], x[1])
                ):
                    f.write(f"{e1}\t{e2}\t{score}\n")

        return bert_int_data, ent_emb_dict, entity_pairs_by_name

    def run_basic_unit(self, kg1, kg2, state):
        bert_int_data = self.convert_data(kg1, kg2, state)
        if self.model_path is not None:
            trained_module = self.__load_model_from_path(self.model_path)
        else:
            trained_module = train_basic_bert(bert_int_data, self.dataset_name)
        ent_emb, entity_pairs, train_candidates, test_candidates = (
            get_entity_embedding_main(
                trained_module,
                bert_int_data.train_ill,
                bert_int_data.test_ill,
                bert_int_data.ent2data,
            )
        )
        ent_emb_dict = self._ent_emb_to_dict(
            kg1, kg2, bert_int_data.index2entity, ent_emb
        )
        return (
            bert_int_data,
            trained_module,
            ent_emb,
            entity_pairs,
            ent_emb_dict,
            train_candidates,
            test_candidates,
        )

    def convert_data(self, kg_l: KG, kg_r: KG, state: AlignmentState) -> BertIntInput:
        logger.info("Converting data...")
        # ent_index(ent_id)2entity / relation_index(rel_id)2relation
        # index2entity = read_id2object([data_path + "ent_ids_1",data_path + "ent_ids_2"])

        for idx, entity in enumerate(kg_l.entity_set | kg_r.entity_set):
            entity.global_entity_id = idx

        index2entity_l = {e.global_entity_id: e.name for e in kg_l.entity_set}
        index2entity_r = {e.global_entity_id: e.name for e in kg_r.entity_set}
        index2entity = {**index2entity_l, **index2entity_r}

        logger.info(f"Index2Entity: {DictUtils.dict_head(index2entity)}")

        # index2rel = read_id2object([data_path + "rel_ids_1",data_path + "rel_ids_2"])

        for idx, entity in enumerate(kg_l.relation_set | kg_r.relation_set):
            entity.global_relation_id = idx

        index2rel_l = {r.global_relation_id: r.name for r in kg_l.relation_set}
        index2rel_r = {r.global_relation_id: r.name for r in kg_r.relation_set}
        index2rel = {**index2rel_l, **index2rel_r}

        logger.info(f"Index2Relation: {DictUtils.dict_head(index2rel)}")

        entity2index = {e: idx for idx, e in index2entity.items()}
        rel2index = {r: idx for idx, r in index2rel.items()}

        logger.info(f"Entity2Index: {DictUtils.dict_head(entity2index)}")
        logger.info(f"Rel2Index: {DictUtils.dict_head(rel2index)}")

        # triples
        # rel_triples_1 = read_idtuple_file(data_path + 'triples_1')
        # rel_triples_2 = read_idtuple_file(data_path + 'triples_2')

        rel_triples_1 = EntityPairUtils.object_relation_tuples_to_index(
            entity2index, rel2index, kg_l.relation_tuple_list
        )
        rel_triples_2 = EntityPairUtils.object_relation_tuples_to_index(
            entity2index, rel2index, kg_r.relation_tuple_list
        )

        logger.info(f"RelTriples1: {ListUtils.list_head(rel_triples_1)}")
        logger.info(f"RelTriples2: {ListUtils.list_head(rel_triples_2)}")

        # index_with_entity_1 = read_idobj_tuple_file(data_path + 'ent_ids_1')
        # index_with_entity_2 = read_idobj_tuple_file(data_path + 'ent_ids_2')

        index_with_entity_1 = [(e.global_entity_id, e.name) for e in kg_l.entity_set]
        index_with_entity_2 = [(e.global_entity_id, e.name) for e in kg_r.entity_set]

        # ill
        # train_ill = read_idtuple_file(data_path + 'sup_pairs')
        # test_ill = read_idtuple_file(data_path + 'ref_pairs')

        train_ill = []
        test_ill = []
        max_train_length = len(state.entity_alignments) * self.training_max_percentage

        for e1, e2, prob in sorted(
            state.entity_alignments, key=lambda x: x[2], reverse=True
        ):
            my_tuple = (
                entity2index[e1] if e1 is not None else None,
                entity2index[e2] if e2 is not None else None,
            )
            if (
                prob >= self.training_threshhold
                and len(train_ill) <= max_train_length
                and all(my_tuple)
            ):
                train_ill.append(my_tuple)
            else:
                test_ill.append(my_tuple)

        if self.debug_file_output_dir:
            with open(f"{self.debug_file_output_dir}/train_ill.csv", "w") as f:
                for e1, e2 in train_ill:
                    f.write(f"{index2entity[e1]}\t{index2entity[e2]}\n")

        if self.gold_result is not None:
            self._show_stats(
                set(map(lambda x: (index2entity[x[0]], index2entity[x[1]]), train_ill)),
                set(self.gold_result),
            )
        else:
            logger.warning("No gold result provided, skipping training stats")

        # with open(f"data/dbp15k/fr_en/converted/ref_pairs", "r") as f:
        #     test_ill = []
        #     import csv
        #     reader = csv.reader(f, delimiter="\t")
        #     for row in reader:
        #         e1, e2 = row
        #         e1, e2 = entity2index[e1], entity2index[e2]
        #         test_ill.append((e1, e2))

        # with open(f"data/dbp15k/fr_en/converted/sup_pairs", "r") as f:
        #     train_ill = []
        #     import csv
        #     reader = csv.reader(f, delimiter="\t")
        #     for row in reader:
        #         e1, e2 = row
        #         e1, e2 = entity2index[e1], entity2index[e2]
        #         train_ill.append((e1, e2))

        ent_ill = []
        ent_ill.extend(train_ill)
        ent_ill.extend(test_ill)

        logger.info(f"TrainILL: {ListUtils.list_head(train_ill)}")
        logger.info(f"TestILL: {ListUtils.list_head(test_ill)}")
        logger.info(f"EntILL: {ListUtils.list_head(ent_ill)}")

        # ent_idx
        entid_1 = [entid for entid, _ in index_with_entity_1]
        # entid_1 = [e.id for e in kg_l.entity_set]

        entid_2 = [entid for entid, _ in index_with_entity_2]
        # entid_2 = [e.id for e in kg_r.entity_set]

        entids = [*entid_1, *entid_2]

        logger.info(f"EntID1: {ListUtils.list_head(entid_1)}")
        logger.info(f"EntID2: {ListUtils.list_head(entid_2)}")
        logger.info(f"EntIDs: {ListUtils.list_head(entids)}")

        # ent2descriptionTokens
        Tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        descriptions_dict_l = self._build_desc_dict_from_attribute_or_name(
            kg_l, self.description_name_l
        )
        descriptions_dict_r = self._build_desc_dict_from_attribute_or_name(
            kg_r, self.description_name_r
        )
        descriptions_dict = {**descriptions_dict_l, **descriptions_dict_r}

        if self.des_dict_path != None:
            import pickle

            from_file = pickle.load(open(self.des_dict_path, "rb"))
            logger.info(f"Ent2DesTokensFromFile: {len(from_file)}")
            descriptions_dict = {**descriptions_dict, **from_file}

        ent2desTokens = ent2desTokens_generateFromDict(
            Tokenizer,
            descriptions_dict,
            [index2entity[id] for id in entid_1],
            [index2entity[id] for id in entid_2],
            debug_file=(
                self.debug_file_output_dir + "/ent2desTokens.csv"
                if self.debug_file_output_dir
                else None
            ),
        )

        logger.info(f"DescriptionsDict: {DictUtils.dict_head(descriptions_dict)}")
        logger.info(f"DescriptionsDict: {DictUtils.dict_tail(descriptions_dict)}")
        logger.info(f"Ent2DesTokens: {DictUtils.dict_head(ent2desTokens)}")

        # ent2basicBertUnit_input.
        ent2tokenids = ent2Tokens_gene(Tokenizer, ent2desTokens, entids, index2entity)
        ent2data = ent2bert_input(entids, Tokenizer, ent2tokenids)

        logger.info(f"Ent2Data: {DictUtils.dict_head(ent2data)}")
        logger.info(f"Ent2TokenIDs: {DictUtils.dict_head(ent2tokenids)}")

        return BertIntInput(
            ent_ill=ent_ill,
            train_ill=train_ill,
            test_ill=test_ill,
            index2rel=index2rel,
            index2entity=index2entity,
            rel2index=rel2index,
            entity2index=entity2index,
            ent2data=ent2data,
            rel_triples_1=rel_triples_1,
            rel_triples_2=rel_triples_2,
        )

    def _build_desc_dict_from_attribute_or_name(
        self, kg: KG, description_name: Optional[str]
    ):
        if description_name is None:
            return {ent.name: get_name(ent.name) for ent in kg.entity_set}
        descriptions_l = [
            attribute_tuple
            for attribute_tuple in kg.attribute_tuple_list
            if attribute_tuple[1].name == description_name
        ]

        descriptions_dict = {ent.name: get_name(ent.name) for ent in kg.entity_set}
        descriptions_dict = {
            **descriptions_dict,
            **{
                ent.name: f"{self._get_name(ent.name)} {val.value}".strip()
                for ent, attr, val in descriptions_l
            },
        }
        return descriptions_dict

    @staticmethod
    def _get_name(name: str) -> str:
        processed_name = get_name(name)
        # if name matches Q\d+ then it is a wikidata entity
        if re.match(r"Q\d+", processed_name):
            return ""

        return processed_name
