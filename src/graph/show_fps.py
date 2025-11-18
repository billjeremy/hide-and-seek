from __future__ import annotations
import os
import shutil
from typing import List, Tuple
from tqdm import trange

from .arbo import TreeHyperparams, TreeGeneratorShow, TreeVisualizer, TreeAnalyzer
from .file_distribution_refactored import FileDistributionHyperparams, FileDistributionGenerator


def run_one_arbo(i: int, arbo_hp: TreeHyperparams, output_path: str, all_paths: List[str]) -> Tuple:
    generator = TreeGeneratorShow(
        arbo_hp.n_nodes[i], arbo_hp.max_children[i], arbo_hp.max_height[i], seed=i
    )
    graph, root = generator.generate_tree_bottom_up(output_path)

    arbo_name = f"arbo{i}_n{arbo_hp.n_nodes[i]}_c{arbo_hp.max_children[i]}_h{arbo_hp.max_height[i]}"
    arbo_path = os.path.join(output_path, arbo_name)
    all_paths.append(arbo_path)

    TreeVisualizer.create_folder_structure(graph, root, os.path.join(arbo_path, "copy1"))
    viz_path = os.path.join(arbo_path, arbo_name)
    TreeVisualizer.generate_graphviz_visualization(graph, root, viz_path)

    return graph, root, arbo_path, arbo_name


def create_copies(arbo_path: str, n_copy: int) -> None:
    copy1_dir = os.path.join(arbo_path, "copy1")
    if not os.path.exists(copy1_dir):
        print(f"   ⚠️  copy1 not found in {arbo_path}")
        return

    for copy_idx in range(2, n_copy + 1):
        copy_dir = os.path.join(arbo_path, f"copy{copy_idx}")
        try:
            shutil.copytree(copy1_dir, copy_dir, dirs_exist_ok=True)
        except Exception as e:
            print(f"   ❌ Error creating copy{copy_idx}: {e}")


def main(n_arbo: int = 5, n_copy: int = 1, output_path: str = 'show_fps_output'):
    os.makedirs(output_path, exist_ok=True)
    arbo_hp = TreeHyperparams(n_arbo)
    for i in range(n_arbo):
        arbo_hp.max_children[i] = 3
        arbo_hp.max_height[i] = 7
    repart_hp = FileDistributionHyperparams(n_copy)

    all_paths = []
    file_dist = FileDistributionGenerator(output_path)
    for i in trange(n_arbo, desc="Generating trees"):
        graph, root, arbo_path, arbo_name = run_one_arbo(i, arbo_hp, output_path, all_paths)
        create_copies(arbo_path, n_copy)
        for j in range(n_copy):
            file_dist.fps_vis(graph, root, arbo_path, arbo_name, repart_hp, j)
            distances, candidates = file_dist.fps_rand_vis(graph, root, arbo_path, arbo_name, repart_hp, j)
            file_dist.rand_vis(graph, root, arbo_path, arbo_name, repart_hp, j)
            file_dist.show_subgraph(graph, root, arbo_path, arbo_name, candidates, distances)


if __name__ == '__main__':
    main()
