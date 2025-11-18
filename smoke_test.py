import os
from graph.arbo import TreeHyperparams, TreeGeneratorShow, TreeVisualizer
from graph.file_distribution_refactored import FileDistributionHyperparams, FileDistributionGenerator


def run_smoke():
    n_arbo = 1
    n_copy = 1
    output_path = "smoke_output"
    os.makedirs(output_path, exist_ok=True)

    arbo_hp = TreeHyperparams(n_arbo)
    # set small fixed params for reproducibility
    for i in range(n_arbo):
        arbo_hp.max_children[i] = 2
        arbo_hp.max_height[i] = 4

    repart_hp = FileDistributionHyperparams(n_copy)

    file_dist = FileDistributionGenerator(output_path)

    # Generate a single tree and process one copy
    for i in range(n_arbo):
        generator = TreeGeneratorShow(arbo_hp.n_nodes[i], arbo_hp.max_children[i], arbo_hp.max_height[i], seed=i)
        graph, root = generator.generate_tree_bottom_up(output_path)
        arbo_name = f"arbo{i}_n{arbo_hp.n_nodes[i]}_c{arbo_hp.max_children[i]}_h{arbo_hp.max_height[i]}"
        arbo_path = os.path.join(output_path, arbo_name)
        # create folder structure for copy1
        TreeVisualizer.create_folder_structure(graph, root, os.path.join(arbo_path, "copy1"))
        # process distribution for copy 0
        file_dist.fps_vis(graph, root, arbo_path, arbo_name, repart_hp, 0)
        distances, candidates = file_dist.fps_rand_vis(graph, root, arbo_path, arbo_name, repart_hp, 0)
        file_dist.rand_vis(graph, root, arbo_path, arbo_name, repart_hp, 0)
        file_dist.show_subgraph(graph, root, arbo_path, arbo_name, candidates, distances)


if __name__ == '__main__':
    run_smoke()
