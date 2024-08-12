import os
import time
import logging
import argparse
import numpy as np

from module.cluster_ea_module import ClusterEAModule
from module.dummy_module import DummyModule
from module.module import AlignmentState, Module
from module.precomputed_embedding_module import PrecomputedEmbeddingModule
from objects.KG import KG
from objects.KGs import KGs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_kg(path_r, path_a=None, sep="\t", name=None):
    kg = KG(name=name)

    if path_a is not None:
        with open(path_r, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = str.strip(line).split(sep=sep)
                if len(params) != 3:
                    print(line)
                    continue
                h, r, t = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_relation_tuple(h, r, t)

        with open(path_a, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                params = str.strip(line).split(sep=sep)
                if len(params) != 3:
                    print(line)
                    continue
                # assert len(params) == 3
                e, a, v = params[0].strip(), params[1].strip(), params[2].strip()
                kg.insert_attribute_tuple(e, a, v)
    else:
        with open(path_r, "r", encoding="utf-8") as f:
            prev_line = ""
            for line in f.readlines():
                params = line.strip().split(sep)
                if len(params) != 3 or len(prev_line) == 0:
                    prev_line += "\n" if len(line.strip()) == 0 else line.strip()
                    continue
                prev_params = prev_line.strip().split(sep)
                e, a, v = (
                    prev_params[0].strip(),
                    prev_params[1].strip(),
                    prev_params[2].strip(),
                )
                prev_line = "".join(line)
                if len(e) == 0 or len(a) == 0 or len(v) == 0:
                    print("Exception: " + e)
                    continue
                if v.__contains__("http"):
                    kg.insert_relation_tuple(e, a, v)
                else:
                    kg.insert_attribute_tuple(e, a, v)
    kg.init()
    kg.print_kg_info()
    return kg


def construct_kgs(dataset_dir, name="KGs", load_chk=None):
    path_r_1 = os.path.join(dataset_dir, "rel_triples_1")
    path_a_1 = os.path.join(dataset_dir, "attr_triples_1")

    path_r_2 = os.path.join(dataset_dir, "rel_triples_2")
    path_a_2 = os.path.join(dataset_dir, "attr_triples_2")

    kg1 = construct_kg(path_r_1, path_a_1, name=str(name + "-KG1"))
    kg2 = construct_kg(path_r_2, path_a_2, name=str(name + "-KG2"))

    kgs = KGs(kg1=kg1, kg2=kg2)
    # load the previously saved PRASE model
    if load_chk is not None:
        kgs.util.load_params(load_chk)
    return kgs


# the balancing function for PRASE
def fusion_func(prob, x, y):
    return 0.8 * prob + 0.2 * np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def run_init_iteration(kgs, ground_truth_path=None):
    kgs.run(test_path=ground_truth_path)


def transform_entity_alignment(entity_alignment: tuple):
    return (
        entity_alignment[0].name if entity_alignment[0] is not None else None,
        entity_alignment[1].name if entity_alignment[1] is not None else None,
        entity_alignment[2],
    )


def run_prase_iteration(
    kgs: KGs,
    embed_module: Module,
    save_dir_path: str,
    embed_module_name: str,
    ground_truth_path=None,
    load_weight=1.0,
    reset_weight=1.0,
    load_ent=True,
    load_emb=True,
    init_reset=False,
    prase_func=None,
):
    save_meantime_result(save_dir_path, kgs, embed_module_name)
    if init_reset is True:
        # load_weight: scale the mapping probability predicted by the PARIS module if loading PRASE from check point
        kgs.util.reset_ent_align_prob(lambda x: reset_weight * x)

    entity_alignments = list(
        map(transform_entity_alignment, kgs.get_all_counterpart_and_prob())
    )
    # entity_alignments, _ = kgs.util.generate_input_for_embed_align(link_path=ground_truth_path)
    alignment_state = embed_module.step(
        kgs.kg_l, kgs.kg_r, AlignmentState(entity_alignments=list(entity_alignments))
    )

    # mapping feedback
    if load_ent is True:
        # ent_links_path = os.path.join(embed_dir, "alignment_results_12")
        # load_weight: scale the mapping probability predicted by the embedding module
        kgs.util.load_ent_links(
            func=lambda x: load_weight * x,
            links=alignment_state.entity_alignments,
            force=True,
        )

    # embedding feedback
    if load_emb is True and alignment_state.entity_embeddings:
        kgs.util.load_embedding(alignment_state.entity_embeddings)

    # set the function balancing the probability (from PARIS) and the embedding similarity
    kgs.set_fusion_func(prase_func)

    # save meantime result before running PARIS again
    # save_meantime_result(save_dir_path, kgs, embed_module_name)

    # test once directly after applying embedding module
    kgs.util.test(
        path=ground_truth_path,
        threshold=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )

    save_meantime_result(save_dir_path, kgs, embed_module_name)
    kgs.run(test_path=ground_truth_path)
    save_meantime_result(save_dir_path, kgs, embed_module_name)


def get_embedding_module(
    save_dir_path: str,
    dataset_name: str,
    ground_truth_path: str,
    training_max_percentage: float,
    interaction_model: bool,
    model_path: str,
    des_dict_path: str,
):
    # embedding_module = PrecomputedEmbeddingModule(
    #     alignments_path=os.path.join(embed_output_path, "alignment_results_12"),
    #     embeddings_path=os.path.join(embed_output_path, "ent_embeds.npy"),
    #     mapping_l_path=os.path.join(embed_output_path, "kg1_ent_ids"),
    #     mapping_r_path=os.path.join(embed_output_path, "kg2_ent_ids")
    # )

    gold_result = set()
    with open(ground_truth_path, "r", encoding="utf8") as f:
        for line in f.readlines():
            params = str.strip(line).split("\t")
            ent_l, ent_r = params[0].strip(), params[1].strip()
            gold_result.add((ent_l, ent_r))

    embedding_module = ClusterEAModule(
        # description_name_1="http://purl.org/dc/elements/1.1/description",
        # description_name_2="http://schema.org/description",
        model_path=model_path,
        training_max_percentage=training_max_percentage,
        debug_file_output_dir=save_dir_path + os.path.join("/clusterea", dataset_name),
        dataset_name=dataset_name,
        gold_result=gold_result,
    )
    # embedding_module = DummyModule()

    logger.info(f"Using {embedding_module.__class__.__name__} as the embedding module")
    return embedding_module


def main():
    args = parse_args()
    base, _ = os.path.split(os.path.abspath(__file__))
    dataset_name = args.dataset_path
    dataset_path = os.path.join(os.path.join(base, "data"), dataset_name)

    print("Construct KGs...")
    # load the KG files from relation and attribute triples to construct the KGs object
    # use load_chk to load the PARIS model from a check point
    # note that, due to the limitation of file size, we do not provide the check point file for performing PRASE
    # surprisingly, it may make the result better than the one reported in the paper
    kgs = construct_kgs(dataset_dir=dataset_path, name=dataset_name, load_chk=None)

    # set the number of processes
    kgs.set_worker_num(10)

    # set the iteration number of PARIS
    kgs.set_iteration(args.iterations)

    # ground truth mapping path
    ground_truth_mapping_path = os.path.join(dataset_path, "ent_links")

    # test the model and show the metrics
    # kgs.util.test(path=ground_truth_mapping_path, threshold=0.1)

    # using the following line of code to run the initial iteration of PRASE (i.e., PARIS, without any feedback)
    # the ground truth path is used to show the metrics during the iterations of PARIS
    run_init_iteration(kgs=kgs, ground_truth_path=ground_truth_mapping_path)

    # run PRASE using both the embedding and mapping feedback

    save_dir_name = "output"
    save_dir_path = os.path.join(os.path.join(base, save_dir_name), dataset_name)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # embed_module_name = "MultiKE"
    module = get_embedding_module(
        save_dir_path,
        dataset_name,
        ground_truth_mapping_path,
        args.training_max_percentage,
        args.interaction_model,
        args.model_path,
        args.des_dict_path,
    )
    embed_module_name = module.__class__.__name__

    run_prase_iteration(
        kgs,
        module,
        save_dir_path,
        embed_module_name,
        prase_func=fusion_func,
        ground_truth_path=ground_truth_mapping_path,
        load_ent=True,
    )


def save_meantime_result(save_dir_path, kgs, embed_module_name):
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # save the check point
    # check_point_dir = os.path.join(save_dir_path, "chk")
    # check_point_name = "PRASE-" + embed_module_name + "@" + time_stamp
    # check_point_file = os.path.join(check_point_dir, check_point_name)
    # kgs.util.save_params(check_point_file)

    # save the mapping result
    result_dir = os.path.join(save_dir_path, "mapping")
    result_file_name = "PRASE-" + embed_module_name + "@" + time_stamp + ".txt"
    result_file = os.path.join(result_dir, result_file_name)
    kgs.util.save_results(result_file)

    # generate the input files (training data) for embedding module
    # input_base = os.path.join(save_dir_path, "embed_input")
    # input_dir_name = "PRASE-" + embed_module_name + "@" + time_stamp
    # input_dir = os.path.join(input_base, input_dir_name)
    # kgs.util.write_input_for_embed_align(link_path=ground_truth_mapping_path, save_dir=input_dir, threshold=0.1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PRASE Python + BERT Interaction Model"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations for PARIS. Default is 10.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="D_W_15K_V2/",
        help="Path to the dataset dir.",
    )

    parser.add_argument(
        "--interaction_model",
        type=bool,
        default=True,
        help="Boolean flag indicating if interaction model should be used. Default is True.",
    )

    parser.add_argument(
        "--training_max_percentage",
        type=float,
        default=0.2,
        help="Maximum percentage of data to be used for training. Default is 0.2.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model file.",
    )
    
    parser.add_argument(
        "--des_dict_path",
        type=str,
        default=None,
        help="Path to the description dictionary file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
