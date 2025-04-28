import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from collections import Counter
from networkx import Graph as NXGraph

matplotlib.use('TkAgg')


def simulate_preferential_attachment(k_initial=10, total_nodes=100, max_new_connections=5, animate=True,
                                     animation_pause=0.001):

    G = nx.complete_graph(k_initial)
    initial_node_set = set(range(k_initial))

    pos = None
    if animate:
        plt.figure(figsize=(12, 10))
        try:
            pos = nx.spring_layout(G, seed=42)
            plt.clf()
            nx.draw(G, pos, with_labels=True, node_color='red', node_size=300, font_size=8, width=0.5,
                    edge_color='gray')
            plt.title(f"Initial {k_initial}-Clique")
            plt.pause(1.0)
        except Exception as e:
            print(f"Initial plotting failed: {e}")
            animate = False

    # add new nodes randomly to nodes with prob based on degree of existing connections
    for i in range(k_initial, total_nodes):
        G.add_node(i)
        existing_nodes_list = list(G.nodes())[:-1]

        # determining num of connections
        num_connections_to_make = 0
        if existing_nodes_list:
            num_connections_to_make = min(random.randint(1, max_new_connections), len(existing_nodes_list))

        targets = []
        if num_connections_to_make > 0:
            node_degree_pairs = list(G.degree(existing_nodes_list))

            current_existing_nodes = [node for node, degree in node_degree_pairs]

            total_degree = sum(degree for node, degree in node_degree_pairs)

            if total_degree > 0:
                probabilities = [degree / total_degree for node, degree in node_degree_pairs]
                # Normalize probabilities to ensure they sum to 1.0, correcting potential float precision issues
                prob_sum = sum(probabilities)
                if not np.isclose(prob_sum, 1.0):
                    probabilities = [p / prob_sum for p in probabilities]

                try:
                    # without replacement
                    targets = np.random.choice(
                        current_existing_nodes,
                        size=num_connections_to_make,
                        replace=False,
                        p=probabilities
                    )

                    targets = list(targets)

                except ValueError as e:
                    # Handle potential errors (e.g., probabilities not summing to 1, empty lists)
                    print(f"Warning: numpy.random.choice failed ({e}). Falling back to uniform sampling.")
                    # Fallback needs the list of nodes
                    if num_connections_to_make <= len(current_existing_nodes):
                        targets = random.sample(current_existing_nodes, k=num_connections_to_make)
                    else:
                        # This case shouldn't happen due to min() earlier, but as safeguard:
                        targets = current_existing_nodes[:]  # Take all available

            else:
                # If total_degree is 0 (only isolated nodes exist), connect uniformly
                if num_connections_to_make <= len(current_existing_nodes):
                    targets = random.sample(current_existing_nodes, k=num_connections_to_make)
                else:
                    targets = current_existing_nodes[:]

            # Add the chosen edges
            for target_node in targets:
                # Ensure target_node is the correct type before adding edge
                # if not isinstance(target_node, int): print(f"Warning: Target node {target_node} is not int")
                G.add_edge(i, target_node)

        # optional animation portion
        if animate and plt and pos is not None:

            try:
                # add new node's position
                new_node_pos = np.random.rand(2)
                if targets:
                    # calculate avg pos of neighbors that already have positions
                    neighbor_positions = [pos[t] for t in targets if t in pos]
                    if neighbor_positions:  # check neighbors for positions
                        avg_pos = np.mean(neighbor_positions, axis=0)
                        new_node_pos = avg_pos + np.random.rand(2) * 0.1  # place near their neighbors
                pos[i] = new_node_pos

                pos = nx.spring_layout(G, pos=pos, fixed=list(range(i)), k=0.3, iterations=20, seed=42)

                plt.clf()
                node_colors = ['red' if node in initial_node_set else 'lightblue' for node in G.nodes()]
                nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, font_size=8, width=0.5,
                        edge_color='gray')
                plt.title(f"Adding Node {i + 1}/{total_nodes} (Connects to {len(targets)} nodes)")
                plt.pause(animation_pause)

            except Exception as e:
                print(f"Animation update failed: {e}. Skipping frame.")

    if animate and plt:
        try:
            plt.clf()
            pos = nx.kamada_kawai_layout(G) if len(G) < 500 else nx.spring_layout(G, seed=42)
            node_colors = ['red' if node in initial_node_set else 'lightblue' for node in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, font_size=8, width=0.5,
                    edge_color='gray')
            plt.title(f"Final Graph ({total_nodes} nodes)")
            plt.show()
        except Exception as e:
            print(f"Final plot failed: {e}")

    if not isinstance(G, NXGraph):
        print(f"Error: Simulation resulted in type {type(G)}, not networkx.Graph.")
        return None, initial_node_set

    return G, initial_node_set


def analyze_degrees(G: NXGraph, k_initial: int, initial_node_set: set):
    if not G or not isinstance(G, NXGraph) or G.number_of_nodes() == 0:
        print("Graph is invalid, empty, or not a NetworkX graph. Cannot analyze.")
        return None

    results = {}
    try:
        degrees = dict(G.degree())
        if not degrees:
            print("Graph has no nodes with degrees.")
            return None

        sorted_nodes_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)

        # top k degrees
        actual_k = min(k_initial, len(sorted_nodes_by_degree))
        top_k_nodes_data = sorted_nodes_by_degree[:actual_k]
        results['top_k_degrees'] = [degree for node, degree in top_k_nodes_data]
        # average degree among these top k nodes
        results['avg_top_k_degree'] = np.mean(results['top_k_degrees']) if results['top_k_degrees'] else 0

        # fraction from original graph
        original_nodes_in_top_k = sum(1 for node, degree in top_k_nodes_data if node in initial_node_set)
        results['fraction_original'] = original_nodes_in_top_k / actual_k if actual_k > 0 else 0

        # lowest degrees
        degree_counts = Counter(degrees.values())
        results['min_degree'] = min(degree_counts.keys()) if degree_counts else 0
        results['num_min_degree_nodes'] = degree_counts.get(results['min_degree'], 0)

        # leaves
        results['num_leaves'] = degree_counts.get(1, 0)

        return results

    except Exception as e:
        print(f"An error occurred during degree analysis: {e}")
        return None

if __name__ == '__main__':
    # --- Parameters ---
    INITIAL_K = 50  # k: Number of initial nodes (Reduced for faster example)
    TOTAL_N = 5000  # Total number of nodes (Reduced for faster example)
    MAX_CONN = 50   # Max number of connections for new nodes
    NUM_RUNS = 50   # Number of simulations to average over
    ANIMATE_GRAPH = False   # Keep False for multiple runs, animate just to show off

    print(f"Starting Batch Simulation:")
    print(f"Parameters: Initial Nodes={INITIAL_K}, Total Nodes={TOTAL_N}, Max New Connections={MAX_CONN}")
    print(f"Number of Runs: {NUM_RUNS}")

    all_results = []
    run_times = []

    # --- Simulation Loop ---
    for run in range(NUM_RUNS):
        print(f"\nStarting Run {run + 1}/{NUM_RUNS}...")
        start_time_run = time.time()

        final_graph, initial_nodes = simulate_preferential_attachment(
            k_initial=INITIAL_K,
            total_nodes=TOTAL_N,
            max_new_connections=MAX_CONN,
            animate=ANIMATE_GRAPH
        )

        end_time_run = time.time()
        run_time = end_time_run - start_time_run
        run_times.append(run_time)
        print(f"Run {run + 1} finished in {run_time:.2f} seconds.")

        analysis_results = analyze_degrees(final_graph, INITIAL_K, initial_nodes)
        if analysis_results:
            all_results.append(analysis_results)
        else:
            print(f"Analysis failed for run {run + 1}.")


    # averaging stuff
    print("\nAveraged Results:")
    total_execution_time = sum(run_times)
    print(f"Total time for {NUM_RUNS} runs: {total_execution_time:.2f} seconds.")
    print(f"Average time per run: {np.mean(run_times):.2f} seconds.")

    if not all_results:
        print("\nNo valid results collected. Cannot calculate averages.")
    else:
        avg_fraction_original = np.mean([res['fraction_original'] for res in all_results])
        avg_num_leaves = np.mean([res['num_leaves'] for res in all_results])
        avg_avg_top_k_degree = np.mean([res['avg_top_k_degree'] for res in all_results])

        avg_max_degree = np.mean([res['top_k_degrees'][0] for res in all_results if res['top_k_degrees']])

        print(f"\nAverage Degree of Top {INITIAL_K} Nodes (per run): {avg_avg_top_k_degree:.2f}")
        print(f"Average Highest Degree Node: {avg_max_degree:.2f}")
        print(f"Average Fraction of Top {INITIAL_K} Nodes from Initial Graph: {avg_fraction_original:.3f}")
        print(f"Average Number of Leaves (Degree 1): {avg_num_leaves:.2f}")