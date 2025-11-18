import os
import random
import csv
import networkx as nx
from collections import defaultdict
import numpy as np
from math import comb
from typing import List, Tuple, Dict, Optional, Set

from graphviz import Digraph
from graphviz import Source
import matplotlib.pyplot as plt

class FileDistributionHyperparams:
    """
    Hyperparameters for positive file distribution with explicit management of clusters, folders, and images.
    - A positive cluster contains at least one positive folder.
    - A positive folder contains at least one positive image.
    """
    def __init__(self, n_copy: int):
        self.n_copy = n_copy
        # Per-copy parameters
        self.n_positive_clusters = []  # number of positive clusters
        self.n_positive_folders = []  # total number of positive folders
        self.folders_per_cluster = []  # list of lists: number of folders per cluster
        self.n_positive_images = []  # total number of positive images
        self.n_negative_images = []
        self.images_per_folder = []  # list of lists: number of images per folder
        self.max_gap = []
        self.mean_expected_distance = []
        # Generate hyperparameters for each copy
        for _ in range(n_copy):
            params = self._generate_random_hyperparams()
            n_clusters = params[0]
            n_folders = params[1]
            n_images = params[2]
            # Distribute folders into clusters (at least 1 per cluster)
            folders_dist = self._random_partition(n_folders, n_clusters)
            # Distribute images into folders (at least 1 per folder)
            images_dist = self._random_partition(n_images, n_folders)
            self.n_positive_clusters.append(n_clusters)
            self.n_positive_folders.append(n_folders)
            self.folders_per_cluster.append(folders_dist)
            self.n_positive_images.append(n_images)
            self.n_negative_images.append(params[3])
            self.images_per_folder.append(images_dist)
            self.max_gap.append(params[4])
            self.mean_expected_distance.append(params[5])

    def _random_partition(self, total: int, n_parts: int) -> List[int]:
        """
        Partition 'total' elements into 'n_parts' parts, each part >= 1.
        Returns a list of size n_parts whose sum is total.
        """
        if n_parts >= total:
            return [1] * n_parts + [0] * (n_parts - total)
        # Place 1 in each part, then distribute the rest
        parts = [1] * n_parts
        rest = total - n_parts
        for _ in range(rest):
            idx = random.randint(0, n_parts - 1)
            parts[idx] += 1
        return parts

    def proba_cluster(self, n=5, r=0.49):
        values = np.arange(1, n+1)
        C = (1 - r) / (1 - r**n)
        probas = [C * r**(i) for i in range(n)]
        x = np.random.choice(values, size=1, p=probas).item()
        return x

    def _generate_random_hyperparams(self) -> Tuple[int, int, int, int, float, float]:
        n_positive_clusters = self.proba_cluster()
        n_positive_folders = random.randint(n_positive_clusters, 15)  # at least 1 folder per cluster
        n_positive_images = random.randint(n_positive_folders*2, 200)  # at least 1 image per folder
        n_negative_images = random.randint(300, 900)
        max_gap = random.randint(2, 5)
        mean_expected_distance = random.uniform(1, max(2.0, max_gap - 2.5))
        return n_positive_clusters, n_positive_folders, n_positive_images, n_negative_images, max_gap, mean_expected_distance

    
class FileDistributionGenerator:
    """Optimized generator for distributing files across tree nodes."""

    def __init__(self, output_dir: str = "trees_with_files"):
        self.output_dir = output_dir
        self.error_log = []

        # caches to avoid recomputation
        self._distances_cache = {}
        self._candidates_cache = {}
        
    def _compute_graph_data(self, graph: nx.DiGraph, root: int) -> Tuple[Dict, Dict]:
        """Compute (and cache) distances and candidate nodes for the graph."""
        
        graph_hash = hash(tuple(sorted(graph.edges())))
        
        if graph_hash in self._distances_cache:
            return self._distances_cache[graph_hash], self._candidates_cache[graph_hash]
        else:
            # compute all-pairs shortest-path lengths
            distances = dict(nx.all_pairs_shortest_path_length(graph.to_undirected()))

            # generate candidate centers using FPS
            candidates = self._farthest_point_sampling(graph, root, distances, 10)  # TODO: tune sample size
            # cache results
            self._distances_cache[graph_hash] = distances
            self._candidates_cache[graph_hash] = candidates
            return distances, candidates
                
    def _compute_graph_data_fps_rand(self, graph: nx.DiGraph, root: int) -> Tuple[Dict, Dict]:
        
        distances = dict(nx.all_pairs_shortest_path_length(graph.to_undirected()))
        candidates = self._farthest_point_sampling_rand(graph, root, distances, 10)  # TODO: tune sample size
        return distances, candidates
        
    def _compute_graph_data_fps_rand(self, graph: nx.DiGraph, root: int) -> Tuple[Dict, Dict]:
        """Compute distances and candidates using randomized FPS sampling."""
        distances = dict(nx.all_pairs_shortest_path_length(graph.to_undirected()))
        candidates = self._farthest_point_sampling_rand(graph, root, distances, 10)  # TODO: tune sample size
        return distances, candidates

    def _compute_graph_data_rand(self, graph: nx.DiGraph, root: int) -> Tuple[Dict, Dict]:
        """Compute distances and candidates using uniform random sampling."""
        distances = dict(nx.all_pairs_shortest_path_length(graph.to_undirected()))
        candidates = self._sampling_rand(graph, root, 10)  # TODO: tune sample size
        return distances, candidates

    def _farthest_point_sampling(self, graph: nx.DiGraph, root : int, distances: Dict, n: int) -> List[int]:
        """Farthest Point Sampling: pick nodes maximizing minimum distance to the set."""
        leaves = [node for node in graph.nodes() if graph.out_degree(node) == 0]
        if not leaves:
            return []
        distances_from_root = nx.single_source_shortest_path_length(graph, root)
        # select leaves at maximum depth
        max_depth = max(distances_from_root[node] for node in leaves)
        deepest_leaves = [node for node in leaves if distances_from_root[node] == max_depth]
        # for reproducibility, pick smallest id
        first = min(deepest_leaves)
        
        selected=[first]
        nodes = list(graph.nodes())
        if n == 1:
            return selected
        while len(selected) < n:
            remaining = [node for node in nodes if node not in selected]
            if not remaining:
                break
            # For each candidate compute the min distance to the selected set
            min_distances = {node: min(distances[node][s] for s in selected) for node in remaining}
            # Pick the node that maximizes this minimum distance
            best_node = max(min_distances, key=min_distances.get)
            selected.append(best_node)
        return selected

    def _farthest_point_sampling_rand(self, graph: nx.DiGraph, root : int, distances: Dict, n: int) -> List[int]:
   
        
        nodes = list(graph.nodes())
        
        first = random.sample(nodes,1)[0]
        
        while first == root:
            first = random.sample(nodes,1)[0]
        
        
        selected=[first]
        if n == 1:
            return selected
        
        while len(selected) < n:
            remaining = [node for node in nodes if node not in selected]
            if not remaining:
                break
            
            # For each candidate compute the min distance to the selected set
            min_distances = {node: min(distances[node][s] for s in selected) for node in remaining}
            # Pick the node that maximizes this minimum distance
            best_node = max(min_distances, key=min_distances.get)
            selected.append(best_node)
        return selected
        
    def _sampling_rand(self, graph: nx.DiGraph, root : int, n: int) -> List[int]:
        selected = []
        nodes = list(graph.nodes())
        while len(selected) < n:
            remaining = [node for node in nodes if node not in selected]
            if not remaining:
                break
            # For each candidate pick a random node (avoid the root)
            new_node = random.sample(remaining, 1)[0]
            while new_node == root:
                new_node = random.sample(remaining, 1)[0]
            selected.append(new_node)
        return selected
        

    def _extract_subgraph_nodes(self, candidate: int, distances: Dict, 
                               max_distance: int) -> List[int]:
        """Return nodes within max_distance from candidate using precomputed distances."""
        subgraph_nodes = [node for node in distances[candidate] if distances[candidate][node] <= max_distance]
        return subgraph_nodes

    def _calculate_average_minimum_distance(self, nodes: List[int], 
                                          distances: Dict) -> float:
        """Compute average of per-node minimum distances to other nodes.

        Returns mean_i min_{j != i} distance(node_i, node_j).
        """
        if len(nodes) <= 1:
            return 0.0
        total = sum(min(distances[node_i][node_j] for node_j in nodes if node_j != node_i) 
                   for node_i in nodes)
        return total / len(nodes)
    

    def _generate_combinations(self, subgraph_nodes: List[int], k: int, m: int) -> List[Tuple[int]]:
        """Generate up to m unique random combinations of k nodes from a list."""
        n = len(subgraph_nodes)
        if n < k:
            return []

        max_possible = comb(n, k)
        true_m = min(m, max_possible)

        seen = set()
        combos = []
        max_attempts = true_m * 10  # pour éviter les boucles infinies

        attempts = 0
        while len(combos) < true_m and attempts < max_attempts:
            combo = tuple(sorted(random.sample(subgraph_nodes, k)))
            if combo not in seen:
                seen.add(combo)
                combos.append(combo)
            attempts += 1

        return combos

    def _find_best_combination(self, candidates: List[int], copy_idx: int,
                              distances: Dict, hyperparams: FileDistributionHyperparams, cluster_k: int,
                              m: int = 100) -> Optional[List[int]]:
        """Find the best folder combination for a given candidate set and cluster.

        This evaluates candidate-centered combinations and returns the one
        whose average minimum pairwise distance is closest to the target.
        """

        nb_folders = hyperparams.folders_per_cluster[copy_idx][cluster_k]
        target_distance = hyperparams.mean_expected_distance[copy_idx]

        best_combination = None
        best_score = float('inf')
        all_combinations = []

        # Generate combinations for each candidate
        for candidate in candidates:
            max_gap = hyperparams.max_gap[copy_idx]
            subgraph_nodes = self._extract_subgraph_nodes(candidate, distances, max_gap)

            # widen search until we have enough nodes
            while len(subgraph_nodes) < nb_folders:
                max_gap += 1
                subgraph_nodes = self._extract_subgraph_nodes(candidate, distances, max_gap)

            other_nodes = [node for node in subgraph_nodes if node != candidate]
            needed = nb_folders - 1

            if needed == 0:
                all_combinations.append((candidate,))
            else:
                combos = self._generate_combinations(other_nodes, needed, m)
                for combo in combos:
                    all_combinations.append((candidate,) + combo)

        # Evaluate combinations and pick the one closest to target distance
        for combination in all_combinations:
            real_distance = self._calculate_average_minimum_distance(list(combination), distances)
            score = abs(target_distance - real_distance)

            if score < best_score:
                best_score = score
                best_combination = list(combination)

        return best_combination, (best_combination[0] if best_combination else None)

    def _create_positive_files(self, nodes: List[int], node_to_path: Dict[int, str],
                     copy_idx: int, hyperparams: FileDistributionHyperparams,
                     cluster_k: int) -> bool:
        """Create positive sample files in each target node folder."""
        try:
            images_per_folder = hyperparams.images_per_folder[copy_idx][cluster_k]
            for node in nodes:
                path = node_to_path[node]
                for j in range(images_per_folder):
                    file_path = os.path.join(path, f"P_{j+1:03d}.txt")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"Positive file {j+1} in node {node}\n")
                        f.write(f"Images in folder: {images_per_folder}\n")
            return True
        except Exception as e:
            print(f"Error creating positive files: {e}")
            return False

    def _create_negative_files(self, nodes: List[int], node_to_path: Dict[int, str],
                     copy_idx: int, hyperparams: FileDistributionHyperparams,
                     graph: nx.DiGraph) -> List[int]:
        """Create negative sample files placed randomly across other folders."""
        try:
            negative_candidates = [node for node in graph.nodes() if node not in nodes]
            nb_negative_files = hyperparams.n_negative_images[copy_idx]
            negative_folders = []
            count_per_folder = []
            for i in range(nb_negative_files):
                node = random.choice(negative_candidates)
                if node not in negative_folders:
                    negative_folders.append(node)
                    count_per_folder.append(1)
                else:
                    index = negative_folders.index(node)
                    count_per_folder[index] += 1
                path = node_to_path[node]
                file_path = os.path.join(path, f"N_{i+1:03d}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Negative file {i+1} in node {node}\n")
            return negative_folders, count_per_folder
        except Exception as e:
            print(f"Error creating negative files: {e}")
            return None, None
        

    def _save_hyperparams(self, copy_path: str, copy_idx: int, hyperparams: FileDistributionHyperparams, 
                     positive_folders, negative_folders, count_per_folder, tree_name: str, 
                     copy_name: str, distances: Dict):
        """Save hyperparameters and computed metrics to CSV.

        Writes general info, per-cluster metrics and inter-cluster statistics.
        """
        hyperparam_file = os.path.join(copy_path, "hyperparams.csv")

        # cluster-level metrics
        cluster_metrics = []
        nb_clusters = hyperparams.n_positive_clusters[copy_idx]

        for cluster_idx in range(nb_clusters):
            cluster_folders = positive_folders[cluster_idx]

            # intra-cluster distances
            intra_cluster_distances = []
            for i, folder1 in enumerate(cluster_folders):
                for j, folder2 in enumerate(cluster_folders):
                    if i != j:
                        intra_cluster_distances.append(distances[folder1][folder2])

            avg_intra_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
            min_intra_distance = min(intra_cluster_distances) if intra_cluster_distances else 0
            max_intra_distance = max(intra_cluster_distances) if intra_cluster_distances else 0
            std_intra_distance = np.std(intra_cluster_distances) if intra_cluster_distances else 0

            # diameter and radius
            cluster_diameter = max_intra_distance
            if len(cluster_folders) > 1:
                centroid = min(cluster_folders, key=lambda x: sum(distances[x][y] for y in cluster_folders))
                cluster_radius = max(distances[centroid][folder] for folder in cluster_folders)
            else:
                centroid = cluster_folders[0]
                cluster_radius = 0

            cluster_metrics.append({
                'cluster_id': cluster_idx,
                'nb_folders': len(cluster_folders),
                'folders': cluster_folders,
                'avg_intra_distance': avg_intra_distance,
                'min_intra_distance': min_intra_distance,
                'max_intra_distance': max_intra_distance,
                'std_intra_distance': std_intra_distance,
                'diameter': cluster_diameter,
                'radius': cluster_radius,
                'centroid': centroid,
                'nb_images': hyperparams.images_per_folder[copy_idx][cluster_idx] if cluster_idx < len(hyperparams.images_per_folder[copy_idx]) else 0
            })

        # inter-cluster distances
        inter_cluster_distances = []
        for i in range(nb_clusters):
            for j in range(i + 1, nb_clusters):
                cluster1_folders = positive_folders[i]
                cluster2_folders = positive_folders[j]

                min_distance = float('inf')
                closest_pair = None
                for folder1 in cluster1_folders:
                    for folder2 in cluster2_folders:
                        dist = distances[folder1][folder2]
                        if dist < min_distance:
                            min_distance = dist
                            closest_pair = (folder1, folder2)

                max_distance = 0
                farthest_pair = None
                for folder1 in cluster1_folders:
                    for folder2 in cluster2_folders:
                        dist = distances[folder1][folder2]
                        if dist > max_distance:
                            max_distance = dist
                            farthest_pair = (folder1, folder2)

                all_inter_distances = [distances[f1][f2] for f1 in cluster1_folders for f2 in cluster2_folders]
                avg_distance = np.mean(all_inter_distances)
                std_distance = np.std(all_inter_distances)

                inter_cluster_distances.append({
                    'cluster_pair': f"{i}-{j}",
                    'min_distance': min_distance,
                    'max_distance': max_distance,
                    'avg_distance': avg_distance,
                    'std_distance': std_distance,
                    'closest_pair': closest_pair,
                    'farthest_pair': farthest_pair
                })

        # global metrics
        all_positive_folders = [folder for cluster in positive_folders for folder in cluster]

        # silhouette-like score
        silhouette_scores = []
        for cluster_idx, cluster_folders in enumerate(positive_folders):
            for folder in cluster_folders:
                if len(cluster_folders) > 1:
                    intra_dist = np.mean([distances[folder][other] for other in cluster_folders if other != folder])
                else:
                    intra_dist = 0

                min_inter_dist = float('inf')
                for other_cluster_idx, other_cluster in enumerate(positive_folders):
                    if other_cluster_idx != cluster_idx:
                        inter_dist = np.mean([distances[folder][other] for other in other_cluster])
                        min_inter_dist = min(min_inter_dist, inter_dist)

                if min_inter_dist != float('inf') and max(intra_dist, min_inter_dist) > 0:
                    silhouette = (min_inter_dist - intra_dist) / max(intra_dist, min_inter_dist)
                    silhouette_scores.append(silhouette)

        avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0

        global_dispersion = np.mean([distances[f1][f2] for f1 in all_positive_folders for f2 in all_positive_folders if f1 != f2])

        # write CSV
        import csv
        with open(hyperparam_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # General info
            writer.writerow(['=== GENERAL INFORMATION ==='])
            writer.writerow(['Tree', tree_name])
            writer.writerow(['Copy', copy_name])
            writer.writerow(['Number of positive clusters', hyperparams.n_positive_clusters[copy_idx]])
            writer.writerow(['Total positive folders', hyperparams.n_positive_folders[copy_idx]])
            writer.writerow(['Total positive images', hyperparams.n_positive_images[copy_idx]])
            writer.writerow(['Total negative images', hyperparams.n_negative_images[copy_idx]])
            writer.writerow(['Number of negative folders', len(negative_folders)])
            writer.writerow(['Max theoretical gap', hyperparams.max_gap[copy_idx]])
            writer.writerow(['Mean expected distance', f"{hyperparams.mean_expected_distance[copy_idx]:.2f}"])
            writer.writerow([])

            # Global metrics
            writer.writerow(['=== GLOBAL METRICS ==='])
            writer.writerow(['Global dispersion', f"{global_dispersion:.2f}"])
            writer.writerow(['Average silhouette score', f"{avg_silhouette:.3f}"])
            writer.writerow(['All positive folders', str(all_positive_folders)])
            writer.writerow([])

            # Per-cluster metrics
            writer.writerow(['=== CLUSTER METRICS ==='])
            writer.writerow(['Cluster_ID', 'Nb_Folders', 'Folders', 'Nb_Images', 'Avg_Intra_Dist', 
                            'Min_Intra', 'Max_Intra', 'Std_Intra', 'Diameter', 'Radius', 'Centroid'])

            for metrics in cluster_metrics:
                writer.writerow([
                    metrics['cluster_id'],
                    metrics['nb_folders'],
                    str(metrics['folders']),
                    metrics['nb_images'],
                    f"{metrics['avg_intra_distance']:.2f}",
                    f"{metrics['min_intra_distance']:.2f}",
                    f"{metrics['max_intra_distance']:.2f}",
                    f"{metrics['std_intra_distance']:.2f}",
                    f"{metrics['diameter']:.2f}",
                    f"{metrics['radius']:.2f}",
                    metrics['centroid']
                ])
            writer.writerow([])

            # Inter-cluster distances
            writer.writerow(['=== INTER-CLUSTER DISTANCES ==='])
            writer.writerow(['Cluster_Pair', 'Min_Dist', 'Max_Dist', 'Avg_Dist', 'Std_Dist', 'Closest_Pair', 'Farthest_Pair'])

            for inter_metrics in inter_cluster_distances:
                writer.writerow([
                    inter_metrics['cluster_pair'],
                    f"{inter_metrics['min_distance']:.2f}",
                    f"{inter_metrics['max_distance']:.2f}",
                    f"{inter_metrics['avg_distance']:.2f}",
                    f"{inter_metrics['std_distance']:.2f}",
                    str(inter_metrics['closest_pair']),
                    str(inter_metrics['farthest_pair'])
                ])
            writer.writerow([])
            
        
        

    # Méthode auxiliaire pour calculer des métriques supplémentaires
    def _calculate_additional_metrics(self, positive_folders, distances: Dict) -> Dict:
        """Compute extra metrics (compactness, separation, density, isolation).

        Returns a dict with arrays for each metric per cluster.
        """
        all_positive_folders = [folder for cluster in positive_folders for folder in cluster]

        metrics = {
            'compactness': [],
            'separation': [],
            'density': [],
            'isolation': []
        }

        # Compactness and density per cluster
        for cluster in positive_folders:
            if len(cluster) > 1:
                intra_distances = [distances[f1][f2] for f1 in cluster for f2 in cluster if f1 != f2]
                avg_intra = np.mean(intra_distances)
                max_intra = max(intra_distances)
                compactness = avg_intra / max_intra if max_intra > 0 else 0
                metrics['compactness'].append(compactness)

                threshold = avg_intra
                connections = sum(1 for d in intra_distances if d <= threshold)
                max_connections = len(cluster) * (len(cluster) - 1)
                density = connections / max_connections if max_connections > 0 else 0
                metrics['density'].append(density)
            else:
                metrics['compactness'].append(1.0)
                metrics['density'].append(1.0)

        # Separation between clusters
        for i, cluster1 in enumerate(positive_folders):
            for j, cluster2 in enumerate(positive_folders):
                if i < j:
                    min_dist = min(distances[f1][f2] for f1 in cluster1 for f2 in cluster2)
                    metrics['separation'].append(min_dist)

        # Isolation of each cluster
        for i, cluster in enumerate(positive_folders):
            other_folders = [f for j, other_cluster in enumerate(positive_folders) if j != i for f in other_cluster]
            if other_folders:
                isolation = np.mean([distances[f1][f2] for f1 in cluster for f2 in other_folders])
                metrics['isolation'].append(isolation)
            else:
                metrics['isolation'].append(float('inf'))

        return metrics

    def process_arborescence(self, graph: nx.DiGraph, root: int, base_path: str, 
                           arbo_name: str, hyperparams: FileDistributionHyperparams, j:int) -> bool:
        """Process a tree and all its copies: select folders and emit files."""
        
        try:
            
            
            # Compute graph data once
            d, c = self._compute_graph_data(graph, root)
            distances = d.copy()
            candidates = c.copy()
            #random.shuffle(candidates)  
            node_to_path = {}

            def map_paths(node, current_path):
                node_to_path[node] = current_path
                for child in graph.successors(node):
                    child_path = os.path.join(current_path, f"folder_{child:03d}")
                    map_paths(child, child_path)

            copy_name = f"copy{j + 1}"
            copy_path = os.path.join(base_path, copy_name)

            root_dir = f"Folder_{root:03d}"
            root_path = os.path.join(copy_path, root_dir)

            map_paths(root, root_path)
            positive_folders = []

            for cluster_k in range(hyperparams.n_positive_clusters[j]):
                best_combination, candidate = self._find_best_combination(candidates, j, distances, hyperparams, cluster_k)
                candidates.remove(candidate)

                if best_combination is None:
                    self.error_log.append({
                        'tree': arbo_name,
                        'copy': copy_name,
                        'error': 'Unable to find a valid combination',
                        'candidates': candidates,
                        'hyperparams': hyperparams.mean_expected_distance[j],
                        'max_gap': hyperparams.max_gap[j]
                    })

                positive_folders.append(best_combination)

                if not self._create_positive_files(best_combination, node_to_path, j, hyperparams, cluster_k):
                    self.error_log.append({
                        'tree': arbo_name,
                        'copy': copy_name,
                        'error': 'Failed to create positive files'
                    })

            positive_folders_flat = sum(positive_folders, [])
            negative_folders, count_per_folder = self._create_negative_files(positive_folders_flat, node_to_path, j, hyperparams, graph)
            self._save_hyperparams(copy_path, j, hyperparams, positive_folders, negative_folders, count_per_folder, arbo_name, copy_name, distances)
            DistributionVisualizer.generate_graphviz_visualization(graph, root, copy_path, j, positive_folders_flat, hyperparams.images_per_folder[j], negative_folders, count_per_folder)
            
            
        except Exception as e:
            print(f"   General error: {e}")
            
            self.error_log.append({
                'tree': arbo_name,
                'copy': 'N/A',
                'error': f'Exception: {str(e)}'
            })

    def fps_vis(self, graph: nx.DiGraph, root: int, base_path: str, 
                           tree_name: str, hyperparams: FileDistributionHyperparams, j:int) -> bool:
      
        
        try:
            
            
            # Compute graph data once
            d, c = self._compute_graph_data(graph, root)
            distances = d.copy()
            candidates = c.copy()
             
            node_to_path = {}
            
            def map_paths(node, current_path):
                node_to_path[node] = current_path
                for child in graph.successors(node):
                    child_path = os.path.join(current_path, f"folder_{child:03d}")
                    map_paths(child, child_path)
                
            copy_name = f"copy{j + 1}_fps"
            copy_path = os.path.join(base_path, copy_name)
                
            root_dir = f"Folder_{root:03d}"
            root_path = os.path.join(copy_path, root_dir)
            
            map_paths(root, root_path)
            
            

            
            DistributionVisualizer.generate_fps_visualization(graph, root, copy_path, j, candidates)
            
            
        except Exception as e:
            print(f"   General error: {e}")
            
            self.error_log.append({
                'tree': tree_name,
                'copy': 'N/A',
                'error': f'Exception: {str(e)}'
            })        
    
    def fps_rand_vis(self, graph: nx.DiGraph, root: int, base_path: str, 
                           arbo_name: str, hyperparams: FileDistributionHyperparams, j:int) -> bool:
      
        
        try:
            
            
            # Compute graph data once
            d, c = self._compute_graph_data_fps_rand(graph, root)
         
            distances = d.copy()
            candidates = c.copy()
             
            node_to_path = {}
            
            def map_paths(node, current_path):
                node_to_path[node] = current_path
                for child in graph.successors(node):
                    child_path = os.path.join(current_path, f"folder_{child:03d}")
                    map_paths(child, child_path)
                
            copy_name = f"copy{j + 1}_fps_random"
            copy_path = os.path.join(base_path, copy_name)
                
            root_dir = f"Folder_{root:03d}"
            root_path = os.path.join(copy_path, root_dir)
            
            map_paths(root, root_path)
            
            

            
            DistributionVisualizer.generate_fps_visualization(graph, root, copy_path, j, candidates) 
            return distances, candidates
            
            
        except Exception as e:
            print(f"   General error: {e}")
            
            self.error_log.append({
                'tree': arbo_name,
                'copy': 'N/A',
                'error': f'Exception: {str(e)}'
            })

    def rand_vis(self, graph: nx.DiGraph, root: int, base_path: str, 
                           arbo_name: str, hyperparams: FileDistributionHyperparams, j:int) -> bool:
      
        
        try:
            
            
            # Compute graph data once
            d, c = self._compute_graph_data_rand(graph, root)
            distances = d.copy()
            candidates = c.copy()
             
            node_to_path = {}
            
            def map_paths(node, current_path):
                node_to_path[node] = current_path
                for child in graph.successors(node):
                    child_path = os.path.join(current_path, f"folder_{child:03d}")
                    map_paths(child, child_path)
                
            copy_name = f"copy{j + 1}_random"
            copy_path = os.path.join(base_path, copy_name)
                
            root_dir = f"Folder_{root:03d}"
            root_path = os.path.join(copy_path, root_dir)
            
            map_paths(root, root_path)
            
            

            
            DistributionVisualizer.generate_fps_visualization(graph, root, copy_path, j, candidates) 
            
            
        except Exception as e:
            print(f"  General error: {e}")
            
            self.error_log.append({
                'tree': arbo_name,
                'copy': 'N/A',
                'error': f'Exception: {str(e)}'
            })        
    def save_error_log(self, log_path: str):
        """Save the error log to CSV."""
        if not self.error_log:
            return

        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['tree', 'copy', 'error', 'candidates', 'hyperparams', 'max_gap']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.error_log)

    def show_subgraph(self, graph, root, arbo_path, arbo_name, candidates, distances):
        subgraphs_nodes = []
        for candidat in candidates:
            subgraphs_nodes.append(self._extract_subgraph_nodes(candidat, distances, 2))
        
        DistributionVisualizer.generate_subgraph_viz(graph, root,arbo_path, arbo_name,candidates, subgraphs_nodes)
        


        
        
    
    
class DistributionVisualizer:
    """Visualizer for distribution outputs (Graphviz exported images)."""
    
    @staticmethod
    def generate_graphviz_visualization(graph, root, copy_path, j, positive_folders_flat, 
                                      images_per_folder, negative_folders, 
                                      count_per_folder):
        """Generate a Graphviz visualization of the tree and file distribution."""
        try:
            output_path = os.path.join(copy_path, f"copy{j + 1}")
            dot = Digraph(comment="Tree with file distribution")
            dot.attr(rankdir='TB')  # Layout top-to-bottom
            dot.attr('node', shape='box', style='rounded,filled')
            dot.attr('edge', color='darkblue')
            
            # Create dictionaries for quick access
            positive_files_count = dict(zip(positive_folders_flat, images_per_folder))
            negative_files_count = dict(zip(negative_folders, count_per_folder))

            # Add nodes with styles based on their contents
            for node in graph.nodes():
                has_positive = node in positive_files_count
                has_negative = node in negative_files_count
                
                if node == root:
                    # Root node in green
                    label = f"Root\\n{node}"
                    if has_positive:
                        label += f"\\n{positive_files_count[node]} +files"
                    if has_negative:
                        label += f"\\n{negative_files_count[node]} -files"
                    dot.node(str(node), label, shape='doublecircle', color='lightgreen')

                elif has_positive and has_negative:
                    label = f"Node {node}\\n{positive_files_count[node]} +files\\n{negative_files_count[node]} -files"
                    dot.node(str(node), label, color='mediumpurple')

                elif has_positive:
                    label = f"Node {node}\\n{positive_files_count[node]} +files"
                    dot.node(str(node), label, color='lightcoral')

                elif has_negative:
                    label = f"Node {node}\\n{negative_files_count[node]} -files"
                    dot.node(str(node), label, color='lightblue')

                else:
                    label = f"Node {node}"
                    dot.node(str(node), label, color='gray')
            
            # Add edges
            for parent, child in graph.edges():
                dot.edge(str(parent), str(child))

            # Render files
            dot.render(output_path, format='dot', cleanup=True)
            dot.render(output_path, format='png', cleanup=True)
            
        except Exception as e:
            print(f"   Error generating visualization: {e}")
    
    @staticmethod
    def generate_cluster_visualization(graph, root, output_path, positive_folders=None, 
                                     images_per_folder=None):
        """Generate a cluster-focused visualization."""
        try:
            dot = Digraph(comment="Positive clusters")
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', style='rounded,filled')
            dot.attr('edge', color='darkblue')
            
            # Define colors for clusters
            cluster_colors = ['lightcoral', 'lightsalmon', 'lightpink', 'mistyrose', 
                            'peachpuff', 'moccasin', 'wheat', 'khaki', 'lightgoldenrodyellow']
            
            # Create a mapping node -> cluster
            node_to_cluster = {}
            if positive_folders:
                for cluster_idx, cluster_dossiers in enumerate(positive_folders):
                    for dossier in cluster_dossiers:
                        node_to_cluster[dossier] = cluster_idx
            
            # Add nodes
            for node in graph.nodes():
                if node == root:
                    dot.node(str(node), f"Root\\n{node}", shape='doublecircle', color='lightgreen')
                elif node in node_to_cluster:
                    cluster_idx = node_to_cluster[node]
                    color = cluster_colors[cluster_idx % len(cluster_colors)]
                    nb_images = images_per_folder[cluster_idx] if images_per_folder else 0
                    label = f"Cluster {cluster_idx}\\nNode {node}\\n{nb_images} images"
                    dot.node(str(node), label, color=color)
                else:
                    dot.node(str(node), f"Node {node}", color='lightgray')
            
            # Add edges with special colors for clusters
            for parent, child in graph.edges():
                if parent in node_to_cluster and child in node_to_cluster:
                    if node_to_cluster[parent] == node_to_cluster[child]:
                        # Arête intra-cluster
                        cluster_idx = node_to_cluster[parent]
                        color = cluster_colors[cluster_idx % len(cluster_colors)]
                        dot.edge(str(parent), str(child), color=color, penwidth='2')
                    else:
                        # Arête inter-cluster
                        dot.edge(str(parent), str(child), color='red', penwidth='3')
                else:
                    # Arête normale
                    dot.edge(str(parent), str(child))
        
            # Render files
            dot.render(output_path + "_clusters", format='dot', cleanup=True)
            dot.render(output_path + "_clusters", format='png', cleanup=True)
     
        except Exception as e:
            print(f"   Error generating cluster visualization: {e}")
    # Visualizer methods for distribution outputs


    def generate_fps_visualization(graph, root, copy_path, j, candidates):
        try:
            output_path = os.path.join(copy_path, f"copy{j + 1}")
            dot = Digraph(comment="Tree with file distribution")
            dot.attr(rankdir='TB')  # Layout top-to-bottom
            dot.attr('node', shape='box', style='rounded,filled')
            dot.attr('edge', color='darkblue')
            
            # Create dictionaries for quick access

            # Add nodes with styles based on their contents
            for node in graph.nodes():
                if node == root:
                    dot.node(str(node), f"Root\\n{node}", shape='doublecircle', color='lightgreen')
                elif node in candidates:
                    label = f"Node {node}\\n (Candidate #{candidates.index(node)+1})"
                    dot.node(str(node), label, color='yellow')
                else:
                    label = f"Node {node}"
                    dot.node(str(node), label, color='gray')
            
            # Add edges
            for parent, child in graph.edges():
                dot.edge(str(parent), str(child))

            # Render files
            dot.render(output_path, format='dot', cleanup=True)
            dot.render(output_path, format='png', cleanup=True)
            
        except Exception as e:
            print(f"   Error generating visualization: {e}")
            
    def generate_subgraph_viz(graph, root, arbo_path, tree_name, candidates, subgraphs_nodes):
        subgraphs_colors = ['lightcoral', 'lightsalmon', 'lightpink', 'mistyrose', 
                            'peachpuff', 'moccasin', 'hotpink','wheat', 'khaki', 'lightgoldenrodyellow']
        try:
            output_path = os.path.join(arbo_path, f"show_subgraph")
            dot = Digraph(comment="Tree with subgraphs")
            dot.attr(rankdir='TB')  # Disposition de haut en bas
            dot.attr('node', shape='box', style='rounded,filled')
            dot.attr('edge', color='darkblue')
            
            # Create dictionaries for quick access

            # Add nodes with styles based on their contents
            for node in graph.nodes():
                label =''
                color=''
                if node == root:
                    dot.node(str(node), f"Root\\n{node}", shape='doublecircle', color='lightgreen')
                elif node in candidates:
                    label = f"Node {node}\\n (Candidate #{candidates.index(node)+1})"
                    for j in range(len(subgraphs_nodes)):
                        if node in subgraphs_nodes[j]:
                            dot.node(str(node), label, color=subgraphs_colors[j])
                            break
                else:
                    label = f"Node {node}"
                    color = 'gray'
                    for j in range(len(subgraphs_nodes)):
                        if node in subgraphs_nodes[j]:
                            color = subgraphs_colors[j]
                            break
                    dot.node(str(node), label, color=color)

            
            # Add edges
            for parent, child in graph.edges():
                dot.edge(str(parent), str(child))

            dot.render(output_path, format='png', cleanup=True)
            
        except Exception as e:
            print(f"   Error generating visualization: {e}")
            
           
   