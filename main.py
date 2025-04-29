import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from collections import Counter
from networkx import Graph as NXGraph
matplotlib.use('TkAgg')


def simulate_preferential_attachment(k_initial=10, total_nodes=100, max_new_connections=5, animate=False, animation_pause=0.1):

    if k_initial < 1:
        raise ValueError("k_initial must be at least 1.")
    if total_nodes <= k_initial:
        if total_nodes == k_initial:
            print(f"Warning: total_nodes ({total_nodes}) == k_initial ({k_initial}). Returning initial graph.")
            G = nx.complete_graph(k_initial)
            initial_node_set = set(range(k_initial))
            return G, initial_node_set
        else:
            raise ValueError(f"Total nodes ({total_nodes}) must be greater than initial nodes ({k_initial}).")
    if max_new_connections < 1:
        raise ValueError("Maximum new connections must be at least 1.")

    G = nx.complete_graph(k_initial)
    initial_node_set = set(range(k_initial))

    # initialize positions for animation
    pos = None
    if animate:
        plt.figure(figsize=(12, 10))
        try:
            # initial layout should be a star graph of sorts in the center with the initial k nodes with k connections
            pos = nx.spring_layout(G, seed=42)
            plt.clf()
            nx.draw(G, pos, with_labels=True, node_color='red', node_size=300, font_size=8, width=0.5, edge_color='gray')
            plt.title(f"Initial {k_initial}-Clique")
            plt.pause(1.0)
        except Exception as e:
             print(f"Initial plotting failed: {e}")
             animate = False

    # each new node has a connection budget that's randomly chosen from 1 - k (inclusive)
    # each new node connects to that amount of connections to nodes in the graph already
    # connections to nodes in the graph are made proportional to the node's existing degree
    # i.e. rich get richer
    for i in range(k_initial, total_nodes):
        G.add_node(i)
        existing_nodes_list = list(G.nodes())[:-1]

        num_connections_to_make = 0
        if existing_nodes_list:
            num_connections_to_make = random.randint(1, max_new_connections)

        targets = []
        if num_connections_to_make > 0:
            # get connection without replacement, each connected node should connect to a new node with prob
            node_degree_pairs = list(G.degree(existing_nodes_list))
            current_existing_nodes = [node for node, degree in node_degree_pairs]
            total_degree = sum(degree for node, degree in node_degree_pairs)

            if total_degree > 0:
                # calculation of probs
                probabilities = [(degree / total_degree) if total_degree > 0 else 0
                                 for node, degree in node_degree_pairs]
                # normalization, was running into issues, sometimes wouldn't add up correctly
                prob_sum = sum(probabilities)
                if not np.isclose(prob_sum, 1.0) and prob_sum > 0:
                    probabilities = [p / prob_sum for p in probabilities]
                elif prob_sum == 0:
                     total_degree = 0

                if total_degree > 0 and np.isclose(sum(probabilities), 1.0):
                    try:
                        targets = np.random.choice(
                            current_existing_nodes,
                            size=num_connections_to_make,
                            replace=False,
                            p=probabilities
                        )
                        targets = list(targets)
                    except ValueError as e:
                        if num_connections_to_make <= len(current_existing_nodes):
                             targets = random.sample(current_existing_nodes, k=num_connections_to_make)
                        else:
                             targets = current_existing_nodes[:] # take all availabe if there's an issue
                else:
                     if num_connections_to_make <= len(current_existing_nodes):
                          targets = random.sample(current_existing_nodes, k=num_connections_to_make)
                     else:
                          targets = current_existing_nodes[:]

            else:
                # if degree is 0, connect them all
                if num_connections_to_make <= len(current_existing_nodes):
                     targets = random.sample(current_existing_nodes, k=num_connections_to_make)
                else:
                     targets = current_existing_nodes[:]

            for target_node in targets:
                G.add_edge(i, target_node)


        # animation, optional and slow - only for show in presentation ;)
        if animate and plt and pos is not None:
            try:
                # attempting to add new node's position based on number of connections and proximity to neighbors
                # average position based on connected neighbors
                new_node_pos = np.random.rand(2)
                if targets:
                   neighbor_positions = [pos[t] for t in targets if t in pos]
                   if neighbor_positions:
                       avg_pos = np.mean(neighbor_positions, axis=0)
                       new_node_pos = avg_pos + np.random.rand(2) * 0.1
                pos[i] = new_node_pos

                pos = nx.spring_layout(G, pos=pos, fixed=list(range(i)), k=0.3, iterations=20, seed=42)

                plt.clf()
                node_colors = ['red' if node in initial_node_set else 'lightblue' for node in G.nodes()]
                nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, font_size=8, width=0.5, edge_color='gray')
                plt.title(f"Adding Node {i+1}/{total_nodes} (Connects to {len(targets)} nodes)")
                plt.pause(animation_pause)

            except Exception as e:
                 print(f"Animation update failed: {e}. Skipping frame.")

    if animate and plt:
        try:
            plt.clf()
            pos = nx.kamada_kawai_layout(G) if len(G) < 500 else nx.spring_layout(G, seed=42)
            node_colors = ['red' if node in initial_node_set else 'lightblue' for node in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, font_size=8, width=0.5, edge_color='gray')
            plt.title(f"Final Graph ({total_nodes} nodes)")
            plt.show()
        except Exception as e:
             print(f"Final plot failed: {e}")
        finally:
             if plt.get_fignums():
                plt.close()


    if not isinstance(G, NXGraph):
        print(f"Error: Simulation resulted in type {type(G)}, not networkx.Graph.")
        return None, initial_node_set

    return G, initial_node_set

def analyze_degrees(G: NXGraph, k_initial_run: int, initial_node_set: set):

    if not G or not isinstance(G, NXGraph) or G.number_of_nodes() == 0:
        print("Graph is invalid, empty, or not a NetworkX graph. Cannot analyze.")
        return None

    results = {}
    try:
        degrees = dict(G.degree())
        if not degrees:
            print("Graph has no nodes with degrees.")
            return None

        # get nodes with top degrees, get top k nodes
        sorted_nodes_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)

        actual_k = min(k_initial_run, len(sorted_nodes_by_degree))

        # if something happens weird, some bs happens
        if actual_k <= 0:
             results['top_k_degrees'] = []
             results['avg_top_k_degree'] = 0
             results['fraction_original'] = 0
        else:
            top_k_nodes_data = sorted_nodes_by_degree[:actual_k]
            results['top_k_degrees'] = [degree for node, degree in top_k_nodes_data]
            # average degree among them
            results['avg_top_k_degree'] = np.mean(results['top_k_degrees']) if results['top_k_degrees'] else 0
            # fraction of them over how many from original
            list_of_original_nodes_in_top_k = sum(1 for node, degree in top_k_nodes_data if node in initial_node_set)
            results['original_nodes_in_top_k'] = list_of_original_nodes_in_top_k
            results['fraction_original'] = list_of_original_nodes_in_top_k / actual_k if actual_k > 0 else 0

        # lowest degrees, most if not all the time 1
        degree_counts = Counter(degrees.values())
        results['degree_counts'] = dict(degree_counts)
        total_degrees = sum(degree * count for degree, count in degree_counts.items())
        total_nodes = sum(count for count in degree_counts.values())
        results['avg_degree'] = total_degrees / total_nodes if total_nodes > 0 else 0
        results['min_degree'] = min(degree_counts.keys()) if degree_counts else 0
        results['num_min_degree_nodes'] = degree_counts.get(results['min_degree'], 0)

        # get the leaves if any
        results['num_leaves'] = degree_counts.get(1, 0)

        return results

    except Exception as e:
        print(f"An error occurred during degree analysis: {e}")
        return None

if __name__ == '__main__':
    INITIAL_K_DEFAULT = 100
    TOTAL_N_DEFAULT = 2000
    NUM_RUNS = 50
    ANIMATE_GRAPH = False # animation slow, just for show (>ᴗ•) !

    RANDOM_N = False   # Set to True to randomize N
    RANDOM_K = False   # Set to True to randomize K

    MIN_N_RANDOM = 500
    MAX_N_RANDOM = 5000
    MIN_K_RANDOM = 5
    MAX_K_RANDOM = 200

    if RANDOM_N:
        print("~~~ Starting Batch Simulation: RANDOM N ~~~")
        print(f"Fixed k = {INITIAL_K_DEFAULT}, Random N Range = [{MIN_N_RANDOM}, {MAX_N_RANDOM}]")
    elif RANDOM_K:
        print("~~~ Starting Batch Simulation: RANDOM k ~~~")
        print(f"Fixed N = {TOTAL_N_DEFAULT}, Random k Range = [{MIN_K_RANDOM}, {MAX_K_RANDOM}]")
    else:
        print("~~~ Starting Batch Simulation: FIXED PARAMETERS ~~~")
        print(f"Fixed k = {INITIAL_K_DEFAULT}, Fixed N = {TOTAL_N_DEFAULT}")

    print(f"Number of Runs: {NUM_RUNS}")

    all_results = []
    run_times = []
    run_parameters = []

    # usually do it for like 15 runs, takes ~ 1 min long on my laptop
    for run in range(NUM_RUNS):
        k = INITIAL_K_DEFAULT
        n = TOTAL_N_DEFAULT

        if RANDOM_N:
            k = INITIAL_K_DEFAULT
            n = random.randint(MIN_N_RANDOM, MAX_N_RANDOM)
            # n is always greater than k
            n = max(n, k + 1)
        elif RANDOM_K:
            n = TOTAL_N_DEFAULT
            current_max_k = min(MAX_K_RANDOM, n - 1)
            if MIN_K_RANDOM >= current_max_k:
                 k = max(1, current_max_k)
            else:
                 k = random.randint(MIN_K_RANDOM, current_max_k)
            k = max(1, k)

        max_conn_run = max(1, k)

        print(f"\n~~~ Starting Run {run + 1}/{NUM_RUNS} ~~~")
        print(f"Parameters for this run: k={k}, N={n}, Max Connections={max_conn_run}")

        run_parameters.append({'k': k, 'n': n, 'max_conn': max_conn_run, 'run_index': run + 1})
        start_time_run = time.time()

        final_graph, initial_nodes = simulate_preferential_attachment(
            k_initial=k,
            total_nodes=n,
            max_new_connections=max_conn_run,
            animate=ANIMATE_GRAPH
        )

        end_time_run = time.time()
        run_time = end_time_run - start_time_run
        run_times.append(run_time)
        print(f"Run {run + 1} finished in {run_time:.2f} seconds.")

        # Analyze for each run, then analyze for all runs
        if isinstance(final_graph, NXGraph):

            analysis_results = analyze_degrees(final_graph, k, initial_nodes)
            if analysis_results:
                print(f"\nAnalysis Results for Run {run + 1} (k={k}, N={n})")

                if RANDOM_K:
                    print(f"  k is {round(((k / n) * 100), 2)}% of N")
                print(f"  Avg Degree of Top {k} Nodes: {analysis_results['avg_top_k_degree']:.2f}")
                if analysis_results['top_k_degrees']:
                     print(f"  Highest Degree Node: {analysis_results['top_k_degrees'][0]}")
                else:
                     print(f"  Highest Degree Node: N/A")
                     if not RANDOM_K:
                         print(f"  k is {round(((k / n) * 100), 2)}% of N")

                print(f"  Fraction of Top {k} Nodes from Initial Graph: {analysis_results['fraction_original']:.3f}")
                print(f"  Lowest Degree: {analysis_results['min_degree']}")
                print(f"  Number of Nodes with Lowest Degree: {analysis_results['num_min_degree_nodes']}")
                print(f"  Number of Leaves (Degree 1): {analysis_results['num_leaves']}")
                print("-" * 50)

                # store all for analyzing later
                analysis_results['params'] = run_parameters[-1]
                all_results.append(analysis_results)
            else:
                print(f"Analysis failed for run {run + 1}.")
        else:
            print(f"Simulation failed for run {run + 1} (returned type: {type(final_graph)}).")


    # overall analysis
    print("\n~~~ Overall Summary ~~~")
    total_execution_time = sum(run_times)
    print(f"Total execution time for {NUM_RUNS} runs: {total_execution_time:.2f} seconds.")
    print(f"Average time per run: {np.mean(run_times):.2f} seconds.")

    if not all_results:
        print("\nNo valid results collected.")
    else:
        num_successful_runs = len(all_results)
        print(f"\nCollected results from {num_successful_runs} successful runs.")

        print("\n~~~ Overall Average Statistics ~~~")
        if RANDOM_N or RANDOM_K:
             rand_string = "N" if RANDOM_N else "K"
             print(f"(Note: These averages are across runs with varying '{rand_string}')")
        else:
             print(f"(Note: These averages are across runs with fixed k={INITIAL_K_DEFAULT}, N={TOTAL_N_DEFAULT})")


        # averages
        original_nodes_in_top_k = [res['original_nodes_in_top_k'] for res in all_results]
        avg_fraction_original = np.mean(original_nodes_in_top_k)
        avg_degree = np.mean([res['avg_degree'] for res in all_results])
        avg_min_degree = np.mean([res['min_degree'] for res in all_results])
        avg_num_min_degree_nodes = np.mean([res['num_min_degree_nodes'] for res in all_results])
        avg_num_leaves = np.mean([res['num_leaves'] for res in all_results])
        avg_avg_top_k_degree = np.mean([res['avg_top_k_degree'] for res in all_results])

        max_degrees_per_run = [res['top_k_degrees'][0] for res in all_results if res.get('top_k_degrees')]
        avg_max_degree = np.mean(max_degrees_per_run) if max_degrees_per_run else 0

        # expected values given top k
        print(f"Counts of Original Nodes in Top K over {NUM_RUNS} runs: {original_nodes_in_top_k}")
        print(f"\nEstimated Expected Value [Fraction Original]: {avg_fraction_original:.4f}")
        print(f"  (Calculated as the average over {num_successful_runs} runs)")

        print(f"\nAverage 'Avg Degree of Top k Nodes' (per run): {avg_avg_top_k_degree:.2f}")
        print(f"Average Highest Degree Node (across runs): {avg_max_degree:.2f}")
        print(f"Average degree of all nodes: {avg_degree:.3f}")
        print(f"Average Lowest Degree (across runs): {avg_min_degree:.2f}")
        print(f"Average Number of Nodes with Lowest Degree (across runs): {avg_num_min_degree_nodes:.2f}")
        print(f"Average Number of Leaves (Degree 1) (across runs): {avg_num_leaves:.2f}")


        # plots based on random variable used
        print("\n--- Generating Plots ---")
        try:
            plot_title_suffix = ""
            # x-axis changes based on random variable to make it make sense
            if RANDOM_N:
                x_values = [res['params']['n'] for res in all_results]
                x_label = "Total Nodes (N)"
                plot_title_suffix = f"vs. Randomized N (Fixed k={INITIAL_K_DEFAULT})"
            elif RANDOM_K:
                x_values = [res['params']['k'] for res in all_results]
                x_label = "Initial Nodes (k)"
                plot_title_suffix = f"vs. Randomized k (Fixed N={TOTAL_N_DEFAULT})"
            else: # Fixed parameters
                x_values = [res['params']['run_index'] for res in all_results]
                x_label = "Run Number"
                plot_title_suffix = f"Variance across Runs (Fixed k={INITIAL_K_DEFAULT}, N={TOTAL_N_DEFAULT})"


            y_leaves = [res['num_leaves'] for res in all_results]
            y_frac_orig = [res['fraction_original'] for res in all_results]
            y_max_degree = [res['top_k_degrees'][0] if res.get('top_k_degrees') else 0 for res in all_results]
            y_avg_top_k = [res['avg_top_k_degree'] for res in all_results]

            fig, axs = plt.subplots(2, 2, figsize=(14, 10)) # Adjusted figure size
            fig.suptitle(f'Simulation Results: {plot_title_suffix}', fontsize=16)

            # different colors
            colors = range(num_successful_runs)

            # Plot 1: Number of Leaves
            axs[0, 0].scatter(x_values, y_leaves, c=colors, cmap='viridis', alpha=0.7)
            axs[0, 0].set_xlabel(x_label)
            axs[0, 0].set_ylabel("Number of Leaves")
            axs[0, 0].set_title("Leaves vs. " + x_label)
            axs[0, 0].grid(True, linestyle='--', alpha=0.5)

            # Plot 2: Fraction Original
            axs[0, 1].scatter(x_values, y_frac_orig, c=colors, cmap='viridis', alpha=0.7)
            axs[0, 1].set_xlabel(x_label)
            axs[0, 1].set_ylabel("Fraction Original in Top k")
            axs[0, 1].set_title("Fraction Original vs. " + x_label)
            axs[0, 1].grid(True, linestyle='--', alpha=0.5)

            # Plot 3: Highest Degree
            axs[1, 0].scatter(x_values, y_max_degree, c=colors, cmap='viridis', alpha=0.7)
            axs[1, 0].set_xlabel(x_label)
            axs[1, 0].set_ylabel("Highest Degree")
            axs[1, 0].set_title("Highest Degree vs. " + x_label)
            axs[1, 0].grid(True, linestyle='--', alpha=0.5)

            # Plot 4: Average Degree of Top k
            axs[1, 1].scatter(x_values, y_avg_top_k, c=colors, cmap='viridis', alpha=0.7)
            axs[1, 1].set_xlabel(x_label)
            axs[1, 1].set_ylabel("Avg Degree of Top k")
            axs[1, 1].set_title("Avg Top k Degree vs. " + x_label)
            axs[1, 1].grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout(rect=(0, 0.03, 1, 0.95))
            plt.show()
            print("Plot displayed.")

        except Exception as e:
            print(f"An error occurred during plotting: {e}")

