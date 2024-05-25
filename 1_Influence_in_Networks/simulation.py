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


def my_create_graph(edges_path: str) -> nx.Graph:
    """
    Creates the Chic Choc social network
    :param edges_path: A csv file that contains information obout "friendships" in the network
    :return: The Chic Choc social network
    """
    edges = pd.read_csv(edges_path).to_numpy()
    stage = np.zeros((len(edges), 1), dtype=int)
    staged_edges = np.concatenate((edges, stage), axis=1)
    net = nx.Graph()
    net.add_weighted_edges_from(staged_edges, weight='stage')
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


def my_change_network(net: nx.Graph, stage: int) -> nx.Graph:
    """
    Gets the network at staged t and returns the network at stage t+1 (stochastic)
    :param stage: int t
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
                    edges_to_add.append((user1, user2, stage))
    net.add_weighted_edges_from(edges_to_add, weight='stage')
    return net


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


def my_buy_products(net: nx.Graph, purchased: set, neighbors_dict, max_stage: int) -> set:
    """
    Gets the network at the beginning of stage t and simulates a purchase round
    :param neighbors_dict: neighbors for each user.
    :param max_stage: last stage to consider
    :param net: The network at stage t
    :param purchased: All the users who bought a doll up to and including stage t-1
    :return: All the users who bought a doll up to and including stage t
    """
    new_purchases = set()
    for user in net.nodes:
        neighborhood = neighbors_dict[user]
        neighborhood_during_stage = set(nbr for nbr in neighborhood if net.edges[(user, nbr)]['stage'] <= max_stage)
        # edges_during_stage = edges_dict[user]['stage'] <= max_stage
        # neighbors_in_stage = set(edges_during_stage[:][1])
        b = len(neighborhood_during_stage.intersection(purchased))
        n = len(neighborhood_during_stage)
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


# Additional functions:
def find_top_5_populars_naive(net):
    id_list = []
    graph_copy = nx.Graph(net)

    for i in range(k):
        top_node = max(graph_copy.degree(node) for node in graph_copy.nodes())
        id_list.append(top_node)
        graph_copy.remove_node(top_node)

    return id_list


def IC_1(G, S):
    X = []
    cost = get_influencers_cost(cost_path, S)

    reachable_nodes = set()
    for s in S:
        reachable_nodes.union(set(nx.shortest_path(G, s).keys()))
    X.append(len(reachable_nodes))

    return np.mean(np.asarray(X)) - cost


def IC_2(g, S, p=0.5, mc=100):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """

    print("Hi from IC")

    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):

        # Simulate propagation process
        new_active, A = S, S
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                # Determine neighbors that become infected
                np.random.seed(i)
                success = np.random.uniform(0, 1, len(set(g.neighbors(node)))) < p
                new_ones += list(np.extract(success, [n for n in g.neighbors(node)]))

            new_active = (set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A = A.union(new_active)

        spread.append(len(A))
        print("finished IC round " + str(i))

    return np.mean(spread)


def IC_3(g, S, mc=10):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """

    result = []
    cost = get_influencers_cost(cost_path, S)

    for i in range(mc):
        purchased = S
        np.random.seed(i)

        purchased = buy_products(g, purchased)
        result.append(len(purchased) - cost)

    return np.mean(result)


def IC_4(g, S, mc=30):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """

    # must change the graph 6 times during hill climbing!

    result = []
    cost = get_influencers_cost(cost_path, S)

    for i in range(mc):
        purchased = S
        np.random.seed(i)

        for p in range(t):
            purchased = buy_products(g, purchased)

        result.append(len(purchased) - cost)

    return np.mean(result)


def IC_5(g, S, neighbors_dict, mc=30):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """

    # must change the graph 6 times during hill climbing!

    result = []
    cost = get_influencers_cost(cost_path, S)

    for i in range(mc):
        purchased = S
        np.random.seed(i)

        for stage in range(t):
            purchased = my_buy_products(g, purchased, neighbors_dict, stage + 1)

        result.append(len(purchased) - cost)

    return np.mean(result)


def init_network(G):
    for i in range(t):
        G = my_change_network(G, i + 1)
        print("change " + str(i + 1) + " finished.")

    all_neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}
    return all_neighbors, G


def reduce_by_mean(G):
    # to improve run time:
    possible_influencer_nodes = []
    mean_degree = np.mean(np.array([G.degree(node) for node in G.nodes()]))
    for node in G.nodes():
        if G.degree(node) > mean_degree:
            possible_influencer_nodes.append(node)
    return possible_influencer_nodes


def hill_climbing(path):
    S = []
    G = my_create_graph(path)

    neighbors_dict, G = init_network(G)
    possible_influencers = reduce_by_mean(G)

    # Hill Climb:
    for i in range(k):
        print("start finding influencer " + str(i + 1))
        max_mv = 0

        if len(S) != 0:
            const_IC_S = IC_5(G, S, neighbors_dict)

        for node in possible_influencers:
            print("checking node " + str(node))
            if len(S) == 0:
                mv = IC_5(G, {node}, neighbors_dict)
            else:
                mv = IC_5(G, set(S + [node]), neighbors_dict) - const_IC_S

            if mv >= max_mv:
                max_mv = mv
                max_node = node
                print("max node is now " + str(max_node))

        print("influencer " + str(i + 1) + " is " + str(max_node))
        S.append(max_node)
        possible_influencers.remove(max_node)

    return S


def short_simulation(G, cost, S):
    purchased = set(S)
    for stage in range(t):
        purchased = buy_products(G, purchased)
    return len(purchased) - cost


# Define the simulated annealing algorithm
def simulated_annealing(G, initial_nodes, initial_node_scores, cost, initial_score=850, temperature=1.0,
                        cooling_rate=0.99, stopping_criteria=0.001, max_iterations=100):
    current_solution = initial_nodes
    node_scores = initial_node_scores
    best_solution = current_solution
    best_obj_val = initial_score
    current_obj_val = best_obj_val
    iteration = 0

    while temperature > stopping_criteria and iteration < max_iterations:

        neighbor_solution = current_solution.copy()

        ## find worst node by ic score:
        min_score = min(node_scores)
        min_index = node_scores.index(min_score)
        min_node = current_solution[min_index]

        ### make a "random" change instead of min_node
        new_node = min_node ## change
        neighbor_solution[min_index] = new_node

        neighbor_obj_val = short_simulation(G, cost, neighbor_solution)
        delta_obj_val = neighbor_obj_val - current_obj_val

        if delta_obj_val > 0:
            current_solution = neighbor_solution
            current_obj_val = neighbor_obj_val
            node_scores = find_all_ic(G, current_solution)

        # If the neighbor solution has a lower objective function value, accept it with a probability that decreases
        # with temperature
        else:
            acceptance_probability = np.exp(delta_obj_val / temperature)
            if random.random() < acceptance_probability:
                current_solution = neighbor_solution
                current_obj_val = neighbor_obj_val
                node_scores = find_all_ic(G, current_solution)
        # Update the best solution and best objective function value if the current solution is better
        if current_obj_val > best_obj_val:
            best_solution = current_solution
            best_obj_val = current_obj_val

        # Reduce the temperature according to the cooling rate
        temperature *= cooling_rate
        # Increase the iteration count
        iteration += 1

    # Return the best solution and best objective function value
    return best_solution, best_obj_val


def outer_annealing(G, initial_nodes, initial_node_scores):
    # G = create_graph(chic_choc_path)
    #
    # for i in range(t):
    #     G = my_change_network(G, i + 1)
    #     print("change " + str(i + 1) + " finished.")
    #
    # initial_nodes = [483, 3830, 2047, 686, 3101]
    # initial_node_scores = [483, 3830, 2047, 686, 3101]  ##change
    cost = get_influencers_cost(cost_path, initial_nodes)

    best_solution, best_obj_val = simulated_annealing(G, initial_nodes, initial_node_scores, cost)

    # Print the best solution and best objective function value
    print("Best solution:", best_solution)
    print("Best objective function value:", best_obj_val)


def find_all_ic(path, influencers):
    ic_scores = []
    G = my_create_graph(path)
    neighbors_dict, G_changed = init_network(G)

    for i in influencers:
        temp_inf = influencers.copy()
        temp_inf.remove(i)
        score = IC_5(G_changed, set(influencers), neighbors_dict, mc=100) - IC_5(G_changed, set(temp_inf),neighbors_dict,  mc=100)
        ic_scores.append(score)
        print("influencer " + str(i) + " scores " + str(score))

    return ic_scores


def check_nodes_by_ic(G, S, node, neighbors_dict, const_ic):
    print("checking node " + str(node))
    mv = IC_5(G, set(S + [node]), neighbors_dict) - const_ic
    return mv
    # if mv >= max_mv:
    #     max_mv = mv
    #     max_node = node
    #     print("max node is now " + str(max_node))


def improve_one_node(path, S):

    G = my_create_graph(path)
    neighbors_dict, G = init_network(G)
    possible_influencers = [node for node in G.nodes()]
    for s in S:
        possible_influencers.remove(s)

    print("start finding improved influencer")
    const_IC_S = IC_5(G, S, neighbors_dict)

    pool = Pool(6)
    dict = {}
    for node in possible_influencers:
        result = pool.apply_async(check_nodes_by_ic, (G, S, node, neighbors_dict, const_IC_S))
        dict[node] = result.get()
    pool.close()
    pool.join()

    max_node = max(dict, key=lambda k: dict[k])

    # for node in possible_influencers:
    #     print("checking node " + str(node))
    #     mv = IC_4(G, set(S + [node])) - const_IC_S
    #
    #     if mv >= max_mv:
    #         max_mv = mv
    #         max_node = node
    #         print("max node is now " + str(max_node))

    print("new influencer is " + str(max_node) + " with ic_score " + str(dict[max_node]))
    return S.append(max_node)


if __name__ == '__main__':
    print("STARTING")

    chic_choc_network = create_graph(chic_choc_path)

    influencers = [483, 3830, 2047, 686, 3101]
    # ic_scores = find_all_ic(chic_choc_path, influencers)
    ic_scores = [251.18999999999994, 173.08000000000004, 152.97000000000003, 147.92000000000007, 141.06999999999994]

    # # try to improve a single node:
    # print(influencers)
    # print(ic_scores)
    # S = influencers.copy()
    # S.remove(3101) #min ic score
    # print("S: " + str(S))
    # new_influencers = improve_one_node(chic_choc_path, S)
    # print(new_influencers)


    # # after running improve a single node on 2047:
    # influencers1 = [483, 3830, 2047, 686, 3101] #mean 100 is
    # ic_scores1 = [173.1, 167.0, 93.63, 218.03, 105.89]
    # influencers2 = [483, 3830, 686, 3101, 2289] #mean600 is 890
    # ic_scores2 = [207.69] ##only 207 is accurate


    
    print(influencers)
    influencers_cost = get_influencers_cost(cost_path, influencers)
    Score = []
    for sim in range(100):
        chic_choc_network = create_graph(chic_choc_path)
        purchased = set(influencers)

        for i in range(6):
            chic_choc_network = change_network(chic_choc_network)
            purchased = buy_products(chic_choc_network, purchased)
            print("finished round", i + 1)

        score = len(purchased) - influencers_cost
        Score.append(score)

        print("scored " + str(score) + " in simulation " + str(sim+1))
        print("avg so far: " + str(sum(Score) / len(Score)))

    print(Score)
    avg = sum(Score) / len(Score)
    print("*** Your average is: " + str(avg) + " ***")
    
    print("*************** Your final score is " + str(score) + " ***************")
