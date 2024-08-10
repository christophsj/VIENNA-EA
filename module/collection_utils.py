from typing import Iterable, Optional
from objects.Entity import Entity
from objects.Relation import Relation


class ListUtils:
    
    @staticmethod
    def list_head(my_list: list, size=5) -> list:
        print_size = min(size, len(my_list))
        return my_list[:print_size]
    
    @staticmethod
    def flatten(xss):
        return [x for xs in xss for x in xs]
    
class DictUtils:
    
    @staticmethod
    def reverse_dict(d):
        return {v: k for k, v in d.items()}
    
    @staticmethod
    def dict_head(dictionary: dict, size=5) -> dict:
        print_size = min(size, len(dictionary))
        return {k: dictionary[k] for k in list(dictionary.keys())[:print_size]}

    @staticmethod
    def dict_tail(dictionary: dict, size=5) -> dict:
        print_size = min(size, len(dictionary))
        return {k: dictionary[k] for k in list(dictionary.keys())[-print_size:]}
    
class EntityPairUtils:
    
    @staticmethod
    def object_relation_tuples_to_index(ent2index: dict, rel2index: dict,
                                              relation_tuple_list: list[tuple[Entity, Relation, Entity]]):
        return [(ent2index[head.name], rel2index[relation.name], ent2index[tail.name])
                for head, relation, tail in relation_tuple_list]
    
    @staticmethod
    def object_triples_to_name_triples(relation_tuple_list: list[tuple[Entity, Relation, Entity]]):
        return [(head.name, relation.name, tail.name)
                for head, relation, tail in relation_tuple_list]
    
        
    @staticmethod
    def take_max_candidate(scores: list[tuple[str, float, int]]) -> tuple[Optional[str], float]:
        max_score = float("-inf")
        result_entity = None
        for e2, score, _ in scores:
            if score > max_score:
                result_entity = e2
                max_score = score
        return result_entity, max_score
    
    @staticmethod
    def merge_entity_pairs(entity_pairs: Iterable[tuple[str, str, float]],
                                            new_entity_pairs: Iterable[tuple[str, str, float]], result_align_threshold: float) -> list[tuple[str, str, float]]:
        entity_pairs_dict = EntityPairUtils._entity_pairs_to_dict(entity_pairs)
        new_entity_pairs_dict = EntityPairUtils._entity_pairs_to_dict(new_entity_pairs)
        entity_pairs_merged_dict = {}
        for e1, e2, prob in new_entity_pairs:
            if prob < result_align_threshold:
                continue
            # if e1 not in entity_pairs_dict:
            #     entity_pairs_merged_dict[e1] = (e2, prob)
            # else:
            #     if prob > entity_pairs_dict[e1][1] and prob > entity_pairs_merged_dict.get(e1, (None, 0))[1]:
            #         entity_pairs_merged_dict[e1] = (e2, prob)
            
            previous_prob = entity_pairs_dict.get(e1, (None, 0))[1]
            new_prob = previous_prob + prob
            new_prob = min(new_prob, 1.0)
            new_prob = max(new_prob, 0.0)
            entity_pairs_merged_dict[e1] = (e2, new_prob)

        for e1, e2, prob in entity_pairs:
            if e1 is None or e2 is None:
                continue
            if e1 not in new_entity_pairs_dict and prob >= result_align_threshold:
                entity_pairs_merged_dict[e1] = (e2, prob)

        return list(map(lambda x: (x[0], x[1][0], x[1][1]), entity_pairs_merged_dict.items()))

    @staticmethod
    def entity_pairs_to_candidate_dict(entity_pairs: list[tuple[str, str, float]]) -> dict[str, list[tuple[str, float]]]:
        result = {}
        for e1, e2, prob in entity_pairs:
            if e1 not in result:
                result[e1] = [(e2, prob)]
            else:
                result[e1].append((e2, prob))
                result[e1] = sorted(result[e1], key=lambda x: x[1], reverse=True)

        return result

    @staticmethod
    def _entity_pairs_to_dict(entity_pairs: Iterable[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
        result = {}
        for e1, e2, prob in entity_pairs:
            result[e1] = e2, prob
            result[e2] = e1, prob

        return result