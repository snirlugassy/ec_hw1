import hw1_part2 as sw
import hw1_part1 as hw1
import networkx as nx
import pandas as pd


def remove_edges_from_csv(G, csv_file):
    edges = pd.read_csv(csv_file)
    unwanted = [tuple(x) for x in edges.to_numpy()]
    G.remove_edges_from(unwanted)
    return G


def main():
    # Edge Filtering
    links_data = sw.ModelData('links_dataset.csv')
    removed_list = sw.competitive_part(links_data,1000)
    sw.write_file_competition(removed_list)

    # Diffusion Test
    users_data = pd.read_csv('infection_information_set.csv')
    for threshold in [0.3, 0.4, 0.5]:
        print("Threshold = ", threshold)
        G = hw1.create_undirected_weighted_graph(links_data, users_data, 'Q3')
        print("connected components before removal: ", nx.number_connected_components(G))
        G.remove_edges_from(removed_list)
        print("connected components after removal: ", nx.number_connected_components(G))
        print("infected nodes = ", hw1.infected_nodes_after_k_iterations(G,6,threshold))

if __name__ == '__main__':
    main()