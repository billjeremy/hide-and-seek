from graph.arbo import TreeHyperparams, TreeGeneratorShow, TreeVisualizer, TreeAnalyzer
from graph.file_distribution_refactored import FileDistributionHyperparams, FileDistributionGenerator
import shutil
import os
import csv
from typing import List, Tuple
from tqdm import trange


def run_one_arbo(i: int, arbo_hp: TreeHyperparams, output_path: str, all_paths: List[str]) -> Tuple:
    """Generate one tree and create its folder structure on disk.

    Returns (graph, root, arbo_path, arbo_name).
    """
    
    
    generator = TreeGeneratorShow(
        arbo_hp.n_nodes[i], arbo_hp.max_children[i], arbo_hp.max_height[i], seed=i
    )
    graph, root = generator.generate_tree_bottom_up(output_path)
    
    # Create directory for this tree
    arbo_name = f"arbo{i}_n{arbo_hp.n_nodes[i]}_c{arbo_hp.max_children[i]}_h{arbo_hp.max_height[i]}"
    arbo_path = os.path.join(output_path, arbo_name)
    all_paths.append(arbo_path)
    
    # Créer la structure de dossiers pour copy1
    TreeVisualizer.create_folder_structure(graph, root, os.path.join(arbo_path, "copy1"))
    
    
    viz_path = os.path.join(arbo_path, arbo_name)
    TreeVisualizer.generate_graphviz_visualization(graph, root, viz_path)
    
    
    
    return graph, root, arbo_path, arbo_name


def create_copies(arbo_path: str, n_copy: int) -> None:
    """Duplicate copy1 into copy2..copyN under the given tree path."""

    copy1_dir = os.path.join(arbo_path, "copy1")

    if not os.path.exists(copy1_dir):
        print(f"Warning: copy1 not found in {arbo_path}")
        return

    for copy_idx in range(2, n_copy + 1):
        copy_dir = os.path.join(arbo_path, f"copy{copy_idx}")
        try:
            shutil.copytree(copy1_dir, copy_dir, dirs_exist_ok=True)
        except Exception as e:
            print(f"Error creating copy{copy_idx}: {e}")


def save_statistics(all_stats: List[dict], output_path: str) -> None:
    """Save statistics list to CSV if any."""
    if not all_stats:
        return

    stats_file = os.path.join(output_path, "tree_statistics.csv")

    try:
        with open(stats_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = all_stats[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)

        print(f"Statistics saved: {stats_file}")
    except Exception as e:
        print(f"Error saving statistics: {e}")


def create_visualizations(all_out_degrees: List[List[int]], all_leaf_depths: List[List[int]], 
                         arbo_names: List[str], output_path: str) -> None:
    """Create comparative visualizations (boxplots) if data available."""
    if not all_out_degrees or not all_leaf_depths:
        return

    print("Generating comparative visualizations...")

    TreeVisualizer.create_boxplots(
        all_out_degrees, arbo_names,
        "Out-degree comparison",
        os.path.join(output_path, "boxplot_out_degrees.png")
    )

    TreeVisualizer.create_boxplots(
        all_leaf_depths, arbo_names,
        "Leaf depth comparison",
        os.path.join(output_path, "boxplot_leaf_depths.png")
    )


def main():
    
    n_arbo = 10
    n_copy = 10
    output_path = "env_test_complet"

    os.makedirs(output_path, exist_ok=True)

    arbo_hp = TreeHyperparams(n_arbo)
    repart_hp = FileDistributionHyperparams(n_copy=n_copy)

    all_stats = []
    all_paths = []
    
    file_dist = FileDistributionGenerator(output_path)
    for i in trange(n_arbo, desc="Generating trees"):
        graph, root, arbo_path, arbo_name = run_one_arbo(i, arbo_hp, output_path, all_paths)

        create_copies(arbo_path, n_copy)
        for j in trange(n_copy, desc="Distributing files"):
            file_dist.process_arborescence(graph, root, arbo_path, arbo_name, repart_hp, j)
   
    save_statistics(all_stats, output_path)

    error_log_path = os.path.join(output_path, "distribution_errors.csv")
    file_dist.save_error_log(error_log_path)

if __name__ == "__main__":
    main()
