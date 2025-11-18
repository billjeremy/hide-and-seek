from arbo import TreeHyperparams, TreeGeneratorShow, TreeVisualizer, TreeAnalyzer
from file_distribution_refactored import FileDistributionHyperparams, FileDistributionGenerator
import shutil
import os
import csv
from typing import List, Tuple
from tqdm import trange


def run_one_arbo(i: int, arbo_hp: TreeHyperparams, output_path: str, all_paths: List[str]) -> Tuple:
    """Generate a tree and create its folder structure."""
    
    
    generator = TreeGeneratorShow(
        arbo_hp.n_nodes[i], arbo_hp.max_children[i], arbo_hp.max_height[i], seed=i
    )
    graph, root = generator.generate_tree_bottom_up(output_path)
    
    # Create the directory for this tree
    arbo_name = f"arbo{i}_n{arbo_hp.n_nodes[i]}_c{arbo_hp.max_children[i]}_h{arbo_hp.max_height[i]}"
    arbo_path = os.path.join(output_path, arbo_name)
    all_paths.append(arbo_path)
    
    # Create folder structure for copy1
    TreeVisualizer.create_folder_structure(graph, root, os.path.join(arbo_path, "copy1"))
    
    # Generate visualization
    viz_path = os.path.join(arbo_path, arbo_name)
    TreeVisualizer.generate_graphviz_visualization(graph, root, viz_path)
    
    
    return graph, root, arbo_path, arbo_name


def create_copies(arbo_path: str, n_copy: int) -> None:
    """Duplicate copy1 into copy2..copyN under the given tree path."""
    
    
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


if __name__ == "__main__":
    n_arbo = 5
    n_copy = 1
    output_path = 'show_fps_output4'
    os.makedirs(output_path, exist_ok=True)
    
    # Generate hyperparameters

    arbo_hp = TreeHyperparams(n_arbo)
    for i in range(n_arbo):
        arbo_hp.max_children[i]=3
        arbo_hp.max_height[i]=7
    print(arbo_hp.max_children, arbo_hp.max_height)
    repart_hp = FileDistributionHyperparams(n_copy)
    
    # Initialiser les structures de données
    all_stats = []
    all_out_degrees = []
    all_leaf_depths = []
    all_paths = []
    
    file_dist = FileDistributionGenerator(output_path)
    for i in trange(n_arbo, desc="Generating trees"):
        graph, root, arbo_path, arbo_name = run_one_arbo(i, arbo_hp, output_path, all_paths)
        # analyze tree for statistics
        create_copies(arbo_path, n_copy)
        for j in range(n_copy):
            
            # Traiter l'arborescence
            file_dist.fps_vis(graph, root, arbo_path, arbo_name, repart_hp, j)
            distances, candidates = file_dist.fps_rand_vis(graph, root, arbo_path, arbo_name, repart_hp, j)
            file_dist.rand_vis(graph, root, arbo_path, arbo_name, repart_hp, j)
            
            file_dist.show_subgraph(graph, root, arbo_path, arbo_name, candidates, distances)
            