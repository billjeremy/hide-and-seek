from __future__ import annotations
import os
import shutil
from typing import List, Tuple
from tqdm import trange

from .arbo import TreeHyperparams, TreeGeneratorShow, TreeVisualizer


def run_one_arbo(i: int, arbo_hp: TreeHyperparams, output_path: str, all_paths: List[str]) -> Tuple:
    """Generate one tree and create its folder structure."""
    arbo_name = f"arbo{i}_n{arbo_hp.n_nodes[i]}_c{arbo_hp.max_children[i]}_h{arbo_hp.max_height[i]}"
    arbo_path = os.path.join(output_path, arbo_name)
    generator = TreeGeneratorShow(
        arbo_hp.n_nodes[i], arbo_hp.max_children[i], arbo_hp.max_height[i], seed=i
    )
    graph, root = generator.generate_tree_bottom_up(arbo_path)

    all_paths.append(arbo_path)

    # Create folder structure for copy1
    TreeVisualizer.create_folder_structure(graph, root, os.path.join(arbo_path, "copy1"))

    return graph, root, arbo_path, arbo_name


def main(n_arbo: int = 5, output_path: str = "arbo_creation") -> None:
    os.makedirs(output_path, exist_ok=True)

    arbo_hp = TreeHyperparams(n_arbo)

    all_paths = []

    for i in trange(n_arbo, desc="Generating trees"):
        run_one_arbo(i, arbo_hp, output_path, all_paths)


if __name__ == "__main__":
    main()
