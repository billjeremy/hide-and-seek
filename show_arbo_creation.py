from graph.arbo import TreeHyperparams, TreeGeneratorShow, TreeVisualizer

import shutil
import os
import csv
from typing import List, Tuple
from tqdm import trange


def run_one_arbo(i: int, arbo_hp: TreeHyperparams, output_path: str, all_paths: List[str]) -> Tuple:
    """Generate one tree and create its folder structure."""
    
    arbo_name = f"arbo{i}_n{arbo_hp.n_nodes[i]}_c{arbo_hp.max_children[i]}_h{arbo_hp.max_height[i]}"
    arbo_path = os.path.join(output_path, arbo_name)
    generator = TreeGeneratorShow(
        arbo_hp.n_nodes[i], arbo_hp.max_children[i], arbo_hp.max_height[i], seed=i
    )
    graph, root = generator.generate_tree_bottom_up(arbo_path)
    
    # Create directory for this tree
    
    all_paths.append(arbo_path)
    
    # Create folder structure for copy1
    TreeVisualizer.create_folder_structure(graph, root, os.path.join(arbo_path, "copy1"))
    
    
    
    
    return graph, root, arbo_path, arbo_name




def main():
    """Main entry point."""
    
    # Configuration
    n_arbo = 5
    n_copy = 10
    output_path = "arbo_creation"
    
    # Créer le répertoire de sortie
    os.makedirs(output_path, exist_ok=True)
    
    # Generate hyperparameters

    arbo_hp = TreeHyperparams(n_arbo)
    
    
    # Initialiser les structures de données
    all_stats = []
    all_out_degrees = []
    all_leaf_depths = []
    all_paths = []
    

    for i in trange(n_arbo, desc="Generating trees"):
        graph, root, arbo_path, arbo_name = run_one_arbo(i, arbo_hp, output_path, all_paths)
        # analyze tree for statistics
        # prepare stats for saving
        
   
    
if __name__ == "__main__":
    main()
