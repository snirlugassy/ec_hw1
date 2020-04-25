import hw1_part2 as sw
import hw1_part1 as hw1
import networkx as nx
import pandas as pd


def main():
    links_data = sw.ModelData('links_dataset.csv')
    users_data = pd.read_csv('infection_information_set.csv')
    G = hw1.create_undirected_weighted_graph(links_data, users_data, 'Q3')
    
    # local_bridges = list(nx.local_bridges(G))
    # local_bridges = sorted(local_bridges, key=lambda x:x[2])
    # ranking = nx.pagerank(G)
    # for bridge in local_bridges:
    #     if bridge[2] is not float('inf'):

    removed_list = sw.competitive_part(links_data,1000)
    print("connected components before removal: ", nx.number_connected_components(G))
    G.remove_edges_from(removed_list)
    print("connected components after removal: ", nx.number_connected_components(G))
    print("infected nodes = ", hw1.infected_nodes_after_k_iterations(G,6,0.5))
    sw.write_file_competition(removed_list)

if __name__ == '__main__':
    main()