import numpy as np
import networkx as nx
import random
import pandas as pd
from multiprocessing import Pool

k = 5
t = 6
p = 0.01

chic_choc_path = '1_Influence_in_Networks/data/chic_choc_data.csv'
cost_path = '1_Influence_in_Networks/data/costs.csv'

def create_graph(edges_path: str) -> nx.Graph:
    """
    Creates the Chic Choc social network
    :param edges_path: A csv file that contains information obout "friendships" in the network
    :return: The Chic Choc social network
    """
    edges = pd.read_csv(edges_path).to_numpy()
    net = nx.Graph()
    net.add_edges_from(edges)
    return net


def change_network(net: nx.Graph) -> nx.Graph:
    """
    Gets the network at staged t and returns the network at stage t+1 (stochastic)
    :param net: The network at staged t
    :return: The network at stage t+1
    """
    edges_to_add = []
    for user1 in sorted(net.nodes):
        for user2 in sorted(net.nodes, reverse=True):
            if user1 == user2:
                break
            if (user1, user2) not in net.edges:
                neighborhood_size = len(set(net.neighbors(user1)).intersection(set(net.neighbors(user2))))
                prob = 1 - ((1 - p) ** (np.log(neighborhood_size))) if neighborhood_size > 0 else 0  # #################
                if prob >= random.uniform(0, 1):
                    edges_to_add.append((user1, user2))
    net.add_edges_from(edges_to_add)
    return net

def change_network2(net: nx.Graph, all_neighbors) -> nx.Graph:
    p_c = 1 - p
    adj_matrix = nx.to_pandas_adjacency(net)
    comman_neigh = adj_matrix.dot(adj_matrix)
    mask = np.triu(np.ones(adj_matrix.shape),k=0) -adj_matrix.multiply(np.triu(np.ones(adj_matrix.shape),k=0)) -np.eye(adj_matrix.shape[0])
    comman_neigh = comman_neigh.multiply(mask)
    prob = (p_c ** np.log((comman_neigh)))
    random_numbers = np.random.uniform(0, 1, comman_neigh.shape)
    to_add = ((1- random_numbers)>prob).astype(int)
    rows, cols = np.where(to_add == 1)
    new_edges =list(zip(rows,cols))
    touched = set(rows).union(cols)
    net.add_edges_from(new_edges)
    for n in touched:
        all_neighbors[n] = set(net.neighbors(n))

    return net, new_edges


def buy_products(net: nx.Graph, purchased: set) -> set:
    """
    Gets the network at the beginning of stage t and simulates a purchase round
    :param net: The network at stage t
    :param purchased: All the users who bought a doll up to and including stage t-1
    :return: All the users who bought a doll up to and including stage t
    """
    new_purchases = set()
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))
        b = len(neighborhood.intersection(purchased))
        n = len(neighborhood)
        prob = b / n
        if prob >= random.uniform(0, 1):
            new_purchases.add(user)

    return new_purchases.union(purchased)


def get_influencers_cost(cost_path: str, influencers: list) -> int:
    """
    Returns the cost of the influencers you chose
    :param cost_path: A csv file containing the information about the costs
    :param influencers: A list of your influencers
    :return: Sum of costs
    """
    costs = pd.read_csv(cost_path)
    return sum(
        [costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in
         influencers])


def simulation(influencers, influencers_cost, all_neighbors):
    G = create_graph(chic_choc_path)
    purchased = set(influencers)

    for i in range(6):
        G, new_edges = change_network2(G, all_neighbors)
        print("finished change")
        purchased = buy_products(G, purchased)

    score = len(purchased) - influencers_cost
    print("scored " + str(score))

    return score


def simulation2(influencers, influencers_cost):
    G = create_graph(chic_choc_path)
    purchased = set(influencers)

    for i in range(6):
        G = change_network(G)
        # print("finished change")
        purchased = buy_products(G, purchased)

    score = len(purchased) - influencers_cost
    print(str(score))

    return score


if __name__ == '__main__':
    print("STARTING")

    influencers = [483, 3830, 2047, 686, 3101]
    print(influencers)
    chic_choc_network = create_graph(chic_choc_path)
    influencers_cost = get_influencers_cost(cost_path, influencers)
    all_neighbors = {n: set(chic_choc_network.neighbors(n)) for n in chic_choc_network.nodes()}

    pool = Pool(8)
    results = []
    for sim in range(80):
        # result = pool.apply_async(simulation, (influencers, influencers_cost, all_neighbors))
        result = pool.apply_async(simulation2, (influencers, influencers_cost))
        results.append(result)
    pool.close()
    pool.join()
    Score = [result.get() for result in results]

    print(Score)
    avg = sum(Score) / len(Score)
    std = np.std(Score)
    print("*** Your average is: " + str(avg) + " ***")
    print("*** The STD is: " + str(std) + " ***")
