import numpy as np
import networkx as nx
import random
import pandas as pd

# FINAL RESULT
influencers = [483, 3830, 2047, 686, 3101]
# ################################

chic_choc_path = '1_Influence_in_Networks/data/chic_choc_data.csv'
cost_path = '1_Influence_in_Networks/data/costs.csv'

def create_graph(edges_path: str) -> nx.Graph:
    """
    Creates the Chic Choc social network
    :param edges_path: A csv file that contains information about "friendships" in the network
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


## Additional functions:
k = 5  # the number of influences we wish to pouched
p = 0.01
n=100 # number of simulations to calculate the expected value of profit for each node

def fast_change_network(net: nx.Graph, all_neighbors, all_nodes) -> nx.Graph:
    """

    Gets the network at staged t and returns the network at stage t+1 (stochastic)
    :param net: The network at staged t
    :param all_neighbors: A dictionary containing for each node in net its neighbors
    :param all_nodes: A set of all the nodes in net
    :return: The network at stage t+1
    """
    new_edges = []
    touched = set()
    p_c = 1 - p  # calculates the complementary of probability p
    for node in all_nodes:
        # find the nodes we don't want to creat a new edge with
        # i.e. the node's neighbors and itself
        neighbors = all_neighbors[node]
        neighbors.add(node)
        for n in (all_nodes - neighbors):
            # calculate the size of the shared neighbourhood of our node and a node we might want to create an edge with
            neighborhood_size = len(neighbors.intersection(all_neighbors[n]))
            if neighborhood_size > 0:
                # calculate the probability an edge will be created
                prob = (p_c ** (np.log(neighborhood_size)))
                if (1 - random.uniform(0.0, 1.0)) > prob:
                    # add the newly created edge to a list of edges will add to net
                    new_edges.append((node, n))
                    # add nodes involved in creating the new edge in order to update their neighbors
                    touched = touched.union([n, node])
    # simultaneously add  all new edges ( time saving)
    net.add_edges_from(new_edges)
    # update the neighbors of all the nodes who had a change
    for n in touched:
        all_neighbors[n] = set(net.neighbors(n))

    return net


def infect_cycle(G: nx.Graph):
    """
     Gets the network at staged t prior to the calculation of the newly infected nodes
     and returns the network at stage t after the infections
    :param G: The network at staged t prior to the infections
    :return: The network at staged t after  the infections
    """
    new_infected = {}
    # creat a df where the value in each cell with index i
    # represent the number of neighbors node number i have
    infected_proba = pd.DataFrame(G.degree())
    infected_proba = infected_proba.set_index(0)
    for node in G.nodes():
        # reset the number of infected neighbors in each univers
        infected_proba['Infby'] = 0
        # for each of the nodes neighbors go through all the universes he is infected in and
        # count the number of neighbors the node have in each universe
        for n in G.neighbors(node):
            for i in G.nodes[n]['Infby']:
                infected_proba.loc[i, 'Infby'] += 1
        # calculte the probaility that the node will be infected in each universe
        infection_threshold = random.uniform(0.0, 1.0)
        infected_proba['proba'] = infected_proba['Infby'] / infected_proba.loc[node, 1]
        # gets the number of the universe in which the node is newly infected
        infected = infected_proba[infected_proba['proba'] > infection_threshold].index
        # creates the set of all the universes the node is corently infected in ( new and old)
        new_infected[node] = infected.union(G.nodes[node]['Infby'])
    # update the network acording to the calculations
    nx.set_node_attributes(G, new_infected, 'Infby')
    return G



def IC(G):
    """
    The function gets a network G , and returns a pandas data frame of the infected node for each node
    :param G: The network at staged 0 we want to work on
    :return: A pandas dataframe of size len(G.nodes) x n where each entry i,j
             represent the set of nodes node i infected in the j-th iteration
    """
    # create data frame for all nodes
    infected = pd.DataFrame(G.nodes)
    infected = infected.set_index(0)
    infected[0] = 0
    # run n simulations
    for s_iter in range(n):
        G = nx.Graph(chic_choc_network)
        colun_name = 'iter' + str(s_iter)
        infected[colun_name] = infected[0].apply(lambda x: set())
        # initialize a dictionary of the neighbors and a set of the nodes
        all_neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}
        all_nodes = set(G.nodes())
        # for each node initialize the list of nodes he infected in a reality where he is the
        # only infected node at t  =0
        x = range(0, nx.number_of_nodes(G))
        x = {i: {i} for i in x}
        nx.set_node_attributes(G, name='Infby', values=x)
        # repeat for 6 stages
        for t in range(6):
            # check how gets infected
            G = infect_cycle(G)
            # create new edges
            G = fast_change_network(G, all_neighbors, all_nodes)
            print("end" + str(t))
        # for each create the set of nodes he manged to infect in the univers he was the only one infected
        # by finding the indexes of the nodes which got infected in that universe
        for node in all_nodes:
            for j in G.nodes[node]['Infby']:
                if j != node:
                    infected.loc[j, colun_name].add(node)
    return infected


def get_top_five(infected, costs, G):
    """

    :param infected: A  pandas dataframe of size len(G.nodes) x n where each entry i,j
                     represent the set of nodes node i infected in the j-th iteration
    :param costs: A pandas dataframe containing the costs of hiring each influncer
    :param G:
    :return:
    """
    k=5
    # saves the df for future uses
    # infected.to_csv('infected_file4.csv', index=True, columns=infected.columns)
    infected = infected.drop(0, axis=1)
    # create a df for future calcultion
    temp_df = pd.DataFrame(G.nodes)
    temp_df = temp_df.set_index(0)
    temp_df['costs'] = costs
    current_ic = 0
    top = []
    # for each influence we want  to find
    for i in range(k):
        # calculate the expected value of the profit
        # the node gaind using MLE
        temp_df['sum'] = infected.apply(lambda x: x.apply(len).sum(), axis=1)  # get the sum of length of each row
        temp_df['sum'] /= n  # make it right
        temp_df['sum'] = temp_df['sum'] - temp_df['costs']
        temp_df['temp_sum'] = temp_df['sum']
        # calculate the gain each node brought relatively to the gain of our current influences without it
        temp_df['sum'] = temp_df['sum'] - current_ic
        max_idx = (temp_df['sum'].idxmax())  # get the index of most profitable node
        current_ic = temp_df['temp_sum'][max_idx]  # update the profit of our new influence group
        top = top + [max_idx]

        # remove the node from are data, so it won't be chosen twice
        current_sets = (infected.loc[max_idx])
        infected.drop(max_idx, axis=0, inplace=True)
        current_cost = temp_df['costs'][max_idx]
        temp_df.drop(max_idx, axis=0, inplace=True)

        # for each node update the nodes he will infect to be the node he infected + the node the top group infected
        # by merging the node's set with the set of the node newly added to top
        infected = infected.apply(merge_and_divide, args=(current_sets, top), axis=1)

        # update costs
        temp_df['costs'] = temp_df['costs'] + current_cost
    return top


def merge_and_divide(row, current_sets, top):
    """
    merges each set in row with each set in current sets cell wise
    :param row: The row we want to use for the merge
    :param current_sets: The row of sets we want to merge wise
    :param top: A list of values we want to delete from the merge
    :return: row after it been merged with current_sets (cell wise) minus the values in top
    """
    combined = row.combine(current_sets, lambda s1, s2: s1.union(s2))
    combined = combined.apply(lambda s1: s1 - set(top + [row.name]))
    return combined


def hill_climbing(chic_choc_network):
    """
    :param chic_choc_network: Path to download our netwok from
    :return: Top k influences to invest in, in our network
    """
    G = nx.Graph(chic_choc_network)
    df = pd.DataFrame(G.nodes)
    df = df.set_index(0)
    df["cost"] = [get_influencers_cost(cost_path, [node]) for node in G.nodes]
    infected = IC(G)
    top = list(get_top_five(infected, df, G))
    return top


def make_table():
    """ used to merge past runings so we could use them aging"""
    infected1 = pd.read_csv('infected_file.csv' ,index_col=0)
    infected1 = infected1.drop(infected1.columns[0], axis=1)
    for col in infected1.columns:
         infected1[col] = infected1[col].apply(lambda x: set(str(x).split(',')))
         infected1 = infected1.rename(columns={col: col+"_1"})

    infected2 = pd.read_csv('infected_file1.csv' ,index_col=0)
    infected2 = infected2.drop(infected2.columns[0], axis=1)
    for col in infected2.columns:
         infected2[col] = infected2[col].apply(lambda x: set(str(x).split(',')))
         infected2 = infected2.rename(columns={col: col + "_2"})
    infected3 = pd.read_csv('infected_file2.csv', index_col=0)
    infected3 = infected3.drop(infected3.columns[0], axis=1)

    for col in infected3.columns:
        infected3[col] = infected3[col].apply(lambda x: set(str(x).split(',')))
        infected3 = infected3.rename(columns={col: col + "_3"})
    infected4 = pd.read_csv('infected_file3.csv' ,index_col=0)
    infected4 = infected4.drop(infected4.columns[0], axis=1)
    for col in infected4.columns:
         infected4[col] = infected4[col].apply(lambda x: set(str(x).split(',')))
         infected4 = infected4.rename(columns={col: col + "_4"})
    # infected5 = pd.read_csv('infected_file4.csv' ,index_col=0)
    # infected5 = infected5.drop(infected5.columns[0], axis=1)
    # for col in infected5.columns:
    #      infected5[col] = infected5[col].apply(lambda x: set(str(x).split(',')))
    result = pd.concat([infected1, infected2, infected3, infected4],axis=1)
    return result


if __name__ == '__main__':
    print("STARTING")

    chic_choc_network = create_graph(chic_choc_path)

    ###
    G = nx.Graph(chic_choc_network)
    influencers = hill_climbing(chic_choc_network)
    ###

    # infected = make_table()
    #influencers=[483, 3830, 2047, 686, 3101]

    print(influencers)
    influencers_cost = get_influencers_cost(cost_path, influencers)
    purchased = set(influencers)
    for i in range(6):
        chic_choc_network = change_network(chic_choc_network)
        purchased = buy_products(chic_choc_network, purchased)
        print("finished round", i + 1)
    score = len(purchased) - influencers_cost
    print("*************** Your final score is " + str(score) + " ***************")


