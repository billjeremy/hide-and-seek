import os 
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import wasserstein_distance
from statistics import mean, median, variance

from graphviz import Digraph
from graphviz import Source


class TreeHyperparams:
    """Hyperparameters for tree generation."""
    def __init__(self, n_arbo, interval_nodes=(50, 200), interval_children=(2, 10), interval_height=(5, 15)):
        self.n_copy = n_arbo
        self.n_nodes = []
        self.max_children = []
        self.max_height = []

        for _ in range(n_arbo):
            self.n_nodes.append(random.randint(50, 200))
            self.max_children.append(random.randint(2, 10))
            self.max_height.append(random.randint(7, 15))


class TreeGenerator:
    """Random tree generator (bottom-up algorithm)."""
    
    def __init__(self, n_nodes, max_children, max_height, seed=None):
        """Initialize the tree generator.

        Args:
            n_nodes (int): target number of nodes in final tree
            max_children (int): maximum number of children for merge nodes
            max_height (int): maximum allowed height
            seed (int, optional): random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.max_children = max_children
        self.max_height = max_height
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_tree_bottom_up(self):
        """Generate a tree using a bottom-up merging process.

        Returns:
            tuple: (networkx.DiGraph, root_node)
        """
        forests = [(i, 0) for i in range(self.n_nodes)]
        graph = nx.DiGraph()

        for node_id in range(self.n_nodes):
            graph.add_node(node_id)

        next_node_id = self.n_nodes

        while len(forests) > 1:
            candidates = [tree for tree in forests if tree[1] < self.max_height]

            if len(candidates) < 2:
                candidates = forests

            if len(candidates) < 2:
                break

            max_merge = min(self.max_children, len(candidates))
            nb_merge = random.randint(2, max_merge)
            to_merge = random.sample(candidates, nb_merge)

            parent_id = next_node_id
            next_node_id += 1
            graph.add_node(parent_id)

            for (root, depth) in to_merge:
                graph.add_edge(parent_id, root)

            new_depth = 1 + max(depth for (_, depth) in to_merge)

            forests = [f for f in forests if f not in to_merge]
            forests.append((parent_id, new_depth))

        if len(forests) > 1:
            final_root = next_node_id
            graph.add_node(final_root)
            for (root, depth) in forests:
                graph.add_edge(final_root, root)
            root = final_root
        else:
            root = forests[0][0]

        current_nodes = len(graph.nodes())
        if current_nodes > self.n_nodes:
            nodes_to_remove = current_nodes - self.n_nodes
            graph, root = self._remove_excess_nodes(graph, root, nodes_to_remove)

        depths = self._calculate_depths(graph, root)
        nx.set_node_attributes(graph, depths, 'depth')

        return graph, root
    
    def _remove_excess_nodes(self, graph, root, num_to_remove):
        """Remove extra nodes randomly while preserving tree structure.

        Returns (graph, root). Attempts to remove `num_to_remove` leaf or degree-1 nodes.
        """
        
        
        for _ in range(num_to_remove):
            removable_nodes = self._find_removable_nodes(graph, root)

            if not removable_nodes:
                print("Warning: cannot remove more nodes without breaking the tree structure")
                break

            node_to_remove = random.choice(removable_nodes)
            graph = self._remove_node_safely(graph, node_to_remove)
        
        return graph, root
    
    def _find_removable_nodes(self, graph, root):
        """Return nodes that can be safely removed (leaf or single-child nodes)."""
        removable = []
        for node in graph.nodes():
            if node == root:
                continue

            if graph.out_degree(node) == 0:
                removable.append(node)
            elif graph.out_degree(node) == 1:
                removable.append(node)

        return removable
    
    def _remove_node_safely(self, graph, node):
        """Remove a node while preserving tree connectivity where possible."""
        if graph.out_degree(node) == 0:
            graph.remove_node(node)
        elif graph.out_degree(node) == 1:
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))

            if parents and children:
                parent = parents[0]
                child = children[0]
                graph.remove_node(node)
                graph.add_edge(parent, child)

        return graph
    
    def _calculate_depths(self, graph, root):
        """Compute depths of all nodes from the root."""
        depths = {}
        
        def dfs(node, depth=0):
            depths[node] = depth
            for child in graph.successors(node):
                dfs(child, depth + 1)
        
        dfs(root)
        return depths

class TreeGeneratorShow:
    """Tree generator variant that supports visualization output."""
    
    def __init__(self, n_nodes, max_children, max_height, seed=None):
        """Initialize the visual tree generator (like ArborescenceGenerator).

        Same parameters as ArborescenceGenerator.
        """
        self.n_nodes = n_nodes
        self.max_children = max_children
        self.max_height = max_height
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_tree_bottom_up(self, output_path):
        """Generate a tree and optionally prepare visualization output files."""
        forest_min = 1 + math.ceil((self.n_nodes - 1) / self.max_children) * (self.max_children - 1)
        forests = [(i, 0) for i in range(forest_min)]
        graph = nx.DiGraph()

        for node_id in range(forest_min):
            graph.add_node(node_id)

        next_node_id = forest_min
        fusion_count = 0

        while len(forests) > 1:
            candidates = [tree for tree in forests if tree[1] < self.max_height - 1]
            if len(candidates) < 2:
                candidates = forests
            if len(candidates) < 2:
                break

            max_merge = min(self.max_children, len(candidates))
            nb_merge = random.randint(2, max_merge)
            to_merge = random.sample(candidates, nb_merge)

            parent_id = next_node_id
            next_node_id += 1
            graph.add_node(parent_id)

            for (root, depth) in to_merge:
                graph.add_edge(parent_id, root)

            new_depth = 1 + max(depth for (_, depth) in to_merge)
            forests = [f for f in forests if f not in to_merge]
            forests.append((parent_id, new_depth))
            fusion_count += 1

        if len(forests) > 1:
            final_root = next_node_id
            graph.add_node(final_root)
            for (root, depth) in forests:
                graph.add_edge(final_root, root)
            root = final_root
        else:
            root = forests[0][0]

        current_nodes = len(graph.nodes())
        if current_nodes > self.n_nodes:
            nodes_to_remove = current_nodes - self.n_nodes
            graph, root = self._remove_excess_nodes(graph, root, nodes_to_remove)

        depths = self._calculate_depths(graph, root)
        return graph, root
    
    def _remove_excess_nodes(self, graph, root, num_to_remove):
        """Remove randomly excess nodes while preserving tree structure."""
        for _ in range(num_to_remove):
            removable_nodes = self._find_removable_nodes(graph, root)
            if not removable_nodes:
                print("Warning: cannot remove more nodes without breaking the tree structure")
                break
            node_to_remove = random.choice(removable_nodes)
            graph = self._remove_node_safely(graph, node_to_remove)
        return graph, root
    
    def _find_removable_nodes(self, graph, root):
        """Return nodes that can be safely removed (leaf or single-child)."""
        removable = []
        for node in graph.nodes():
            if node == root:
                continue
            if graph.out_degree(node) == 0 or graph.out_degree(node) == 1:
                removable.append(node)
        return removable
    
    def _remove_node_safely(self, graph, node):
        """Remove node while reconnecting parent to child if needed."""
        if graph.out_degree(node) == 0:
            graph.remove_node(node)
        elif graph.out_degree(node) == 1:
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))
            if parents and children:
                parent = parents[0]
                child = children[0]
                graph.remove_node(node)
                graph.add_edge(parent, child)
        return graph
    
    def _calculate_depths(self, graph, root):
        """Compute node depths from the root using DFS."""
        depths = {}
        
        def dfs(node, depth=0):
            depths[node] = depth
            for child in graph.successors(node):
                dfs(child, depth + 1)
        
        dfs(root)
        return depths

class TreeAnalyzer:
    """Statistical analyzer for generated trees."""
    
    @staticmethod
    def analyze_graph(graph):
        """Analyze basic statistics of the tree.

        Returns:
            tuple: (stats_dict, out_degrees_list, leaf_depths_list)
        """
        if not nx.is_weakly_connected(graph):
            print("Warning: graph is not weakly connected")

        out_degrees = [graph.out_degree(node) for node in graph.nodes()]
        depths = nx.get_node_attributes(graph, 'depth')
        missing_depths = [node for node in graph.nodes() if node not in depths]
        if missing_depths:
            print(f"Warning: nodes without depth: {missing_depths}")

        leaves = [node for node in graph.nodes() if graph.out_degree(node) == 0]
        leaf_depths = [depths[node] for node in leaves if node in depths]

        stats = {
            'out_degrees_stats': TreeAnalyzer._compute_descriptive_stats(out_degrees),
            'leaf_depths_stats': TreeAnalyzer._compute_descriptive_stats(leaf_depths),
            'n_nodes': len(graph.nodes()),
            'n_edges': len(graph.edges()),
            'n_leaves': len(leaves),
            'height': max(depths.values()) if depths else 0,
            'is_connected': nx.is_weakly_connected(graph)
        }

        return stats, out_degrees, leaf_depths
    
    @staticmethod
    def _compute_descriptive_stats(values):
        """Return basic descriptive statistics for a list of values."""
        if not values:
            return {
                'mean': 0, 'median': 0, 'variance': 0,
                'q1': 0, 'q2': 0, 'q3': 0, 'min': 0, 'max': 0
            }

        return {
            'mean': mean(values),
            'median': median(values),
            'variance': variance(values) if len(values) > 1 else 0,
            'q1': np.percentile(values, 25),
            'q2': np.percentile(values, 50),
            'q3': np.percentile(values, 75),
            'min': min(values),
            'max': max(values)
        }
    
    @staticmethod
    def compute_distribution_similarity(dist1, dist2):
        """Compute Wasserstein distance between two distributions represented as value lists."""
        if not dist1 or not dist2:
            return float('inf')

        all_values = list(set(dist1 + dist2))
        if not all_values:
            return 0.0

        max_val = max(all_values)
        bins = range(max_val + 2)

        hist1, _ = np.histogram(dist1, bins=bins, density=True)
        hist2, _ = np.histogram(dist2, bins=bins, density=True)

        return wasserstein_distance(hist1, hist2)


class TreeVisualizer:
    @staticmethod
    def generate_graphviz_general(graph, output_path):
        """Generate a Graphviz visualization for a directed graph (non-rooted)."""
        try:
            dot = Digraph(comment="General graph")
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', style='rounded,filled')
            dot.attr('edge', color='darkblue')

            for node in graph.nodes():
                in_deg = graph.in_degree(node)
                out_deg = graph.out_degree(node)
                if out_deg == 0:
                    # leaf
                    dot.node(str(node), f"Leaf\n{node}", color='lightblue')
                else:
                    # internal node
                    dot.node(str(node), f"Node\n{node}", color='lightgreen')

            for parent, child in graph.edges():
                dot.edge(str(parent), str(child))

            dot.render(output_path, format='dot', cleanup=True)
            dot.render(output_path, format='png', cleanup=True)
        except Exception as e:
            print(f"Error generating general visualization: {e}")
    """Visualizer utilities for trees."""
    @staticmethod
    def generate_graphviz_show_visualization(graph, root, output_path):
        """Generate a Graphviz visualization of a tree (rooted)."""
        try:
            dot = Digraph(comment="Tree bottom-up")
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', style='rounded,filled')
            dot.attr('edge', color='darkblue')

            for node in graph.nodes():
                if node == root:
                    dot.node(str(node), f"Root\\n{node}", shape='doublecircle', color='lightcoral')
                elif graph.out_degree(node) == 0:
                    dot.node(str(node), f"Leaf\\n{node}", color='lightblue')
                else:
                    color = 'lightgreen'
                    dot.node(str(node), f"Node\\n{node}", color=color)

            for parent, child in graph.edges():
                dot.edge(str(parent), str(child))

            dot.render(output_path, format='dot', cleanup=True)
            dot.render(output_path, format='png', cleanup=True)
        except Exception as e:
            print(f"Error generating visualization: {e}")

    @staticmethod
    def create_folder_structure(graph, root_node, base_path):
        """Create folder structure on disk matching the tree. Returns mapping node->path."""
        paths = {root_node: os.path.join(base_path, f"Folder_{root_node:03d}")}

        def create_recursive(node):
            path = paths[node]
            os.makedirs(path, exist_ok=True)
            for child in graph.successors(node):
                folder_name = f"folder_{child:03d}"
                child_path = os.path.join(path, folder_name)
                paths[child] = child_path
                create_recursive(child)

        os.makedirs(base_path, exist_ok=True)
        os.makedirs(os.path.join(base_path, f"Folder_{root_node:03d}"), exist_ok=True)
        create_recursive(root_node)
        return paths
    
    @staticmethod
    def generate_graphviz_visualization(graph, root, output_path):
        """Generate a Graphviz visualization of the tree with depth-based styling."""
        try:
            dot = Digraph(comment="Tree bottom-up")
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', style='rounded,filled')
            dot.attr('edge', color='darkblue')

            depths = nx.get_node_attributes(graph, 'depth')
            max_depth = max(depths.values()) if depths else 0

            for node in graph.nodes():
                if node == root:
                    dot.node(str(node), f"Root\\n{node}", shape='doublecircle', color='lightcoral')
                elif graph.out_degree(node) == 0:
                    dot.node(str(node), f"Leaf\\n{node}", color='lightgreen')
                else:
                    depth = depths.get(node, 0)
                    if max_depth > 0:
                        intensity = 1.0 - (depth / max_depth)
                        if intensity > 0.7:
                            color = 'lightblue'
                        elif intensity > 0.4:
                            color = 'lightsteelblue'
                        else:
                            color = 'lightskyblue'
                    else:
                        color = 'lightblue'

                    dot.node(str(node), f"Node\\n{node}", color=color)

            for parent, child in graph.edges():
                dot.edge(str(parent), str(child))

            dot.render(output_path, format='dot', cleanup=True)
            dot.render(output_path, format='png', cleanup=True)
        except Exception as e:
            print(f"Error generating visualization: {e}")
    

    @staticmethod        
    def render_dot_file(dot_file_path, output_format="png"):
        """Open a .dot file and render it to an output format (png, pdf, ...)."""
        try:
            # Lire le contenu du fichier DOT
            with open(dot_file_path, "r", encoding="utf-8") as f:
                dot_source = f.read()
            
            # Créer un objet Graphviz à partir du contenu
            graph = Source(dot_source)
            
            # Rendre le fichier dans le format souhaité (dans le même dossier)
            # output_path = dot_file_path.rsplit(".", 1)[0]  # sans extension
            # graph.render(output_path, format=output_format, cleanup=False)
            
            # print(f"   Rendered successfully: {output_path}.{output_format}")

        except Exception as e:
            print(f"   Error rendering DOT file: {e}")
    
    @staticmethod
    def create_boxplots(data_lists, labels, title, save_path):
        """Create boxplots to compare distributions.

        Args:
            data_lists (list): list of data lists
            labels (list): labels for each distribution
            title (str): plot title
            save_path (str): output file path
        """
        if not data_lists or not all(data_lists):
            print(f"   Not enough data to create boxplot: {title}")
            return
        
        plt.figure(figsize=(12, 7))
        box_plot = plt.boxplot(data_lists, labels=labels, patch_artist=True)
        
        # Personnaliser les couleurs
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_lists)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Values')
        plt.xlabel('Trees')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Boxplot saved: {save_path}")
        except Exception as e:
            print(f"Error saving boxplot: {e}")
            plt.close()


