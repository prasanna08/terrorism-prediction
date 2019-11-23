import pandas as pd
from collections import defaultdict

def read_ty2001_06_dataset():
    '''Read dataset from file.
    
    Returns:
    DataFrame. Pandas dataframe containing data from file.
    '''
    perpetrators = pd.read_csv("./data/names_of_perpetrators.csv")
    terror_plots = pd.read_csv("./data/terror_plots.csv")
    return perpetrators, terror_plots

def get_hyperedges_from_ty2001_06_dataset(perpetrators, terror_plots):
    '''Create a list of hyperedges from the data. Each hyperedge is a frozenset
    of the nodes it contains.

    Returns:
    hyperedges: List. List of all hyperedges
    hyperedges_timestamps: List. Timestamp associated with each hyperedge.
        Timestamps should be in same order as that of hyperedges.
    total_nodes: Int. Total number of nodes in hypergraph.
    '''
    plots = set(
        perpetrators['terror_plot'].values.tolist() +
        perpetrators['terror_plot_2'].dropna().values.tolist())
    plot_to_id = {p: i for i, p in enumerate(plots)}

    plot_with_perpetrator = (
        perpetrators[['person_ID', 'terror_plot']].dropna().values.tolist() +
        perpetrators[['person_ID', 'terror_plot_2']].dropna().values.tolist())
    
    plot_to_perpetrator = defaultdict(set)
    for pid, plot in plot_with_perpetrator:
        plot_to_perpetrator[plot_to_id[plot]].add(pid)
    
    states = (
        perpetrators['last_residency_state'].dropna().values.tolist() +
        perpetrators['state_charged'].dropna().values.tolist())
    state_to_id = {s: i for i, s in enumerate(set(states))}
    
    perpetrator_with_res_state = perpetrators[
        ['person_ID', 'last_residency_state']].dropna().values.tolist()
    res_state_to_perpetrator = defaultdict(set)
    for pid, state in perpetrator_with_res_state:
        res_state_to_perpetrator[state_to_id[state]].add(pid)

    # perpetrator_with_charged_state = perpetrators[
    #     ['person_ID', 'state_charged']].dropna().values.tolist()
    # charged_state_to_perpetrator = defaultdict(set)
    # for pid, state in perpetrator_with_charged_state:
    #     charged_state_to_perpetrator[state_to_id[state]].add(pid)    

    plots = list(plot_to_perpetrator.keys())
    hyperedges = []
    for k in plots:
        hyperedges.append(set(plot_to_perpetrator[k]))
    for k, v in res_state_to_perpetrator.items():
        hyperedges.append(set(v))
    # for k, v in charged_state_to_perpetrator.items():
    #     hyperedges.append(set(v))
    
    plots_with_year = terror_plots[['name', 'year']].fillna(0).values.tolist()
    plot_to_year = {}
    for plot, year in plots_with_year:
        if plot in plot_to_id:
            plot_to_year[plot_to_id[plot]] = year
    
    hyperedges_timestamps = []
    for k in plots:
        hyperedges_timestamps.append(plot_to_year.get(k, 0))

    for i in range(len(res_state_to_perpetrator)):
        hyperedges_timestamps.append(0)
    
    total_nodes = reduce(lambda x, y: x.union(y), hyperedges)
    node_to_id = {node: idx for idx, node in enumerate(total_nodes)}
    for idx in range(len(hyperedges)):
        hedge = hyperedges[idx]
        hyperedges[idx] = frozenset({node_to_id[node] for node in hedge})
    
    hyperedges_to_keep = [
        idx for idx, hedge in enumerate(hyperedges) if len(hedge) > 1]
    hyperedges = [hyperedges[idx] for idx in hyperedges_to_keep]
    hyperedges_timestamps = [
        hyperedges_timestamps[idx] for idx in hyperedges_to_keep]
    
    return hyperedges, hyperedges_timestamps, len(node_to_id)

def read_chsim_dataset():
    '''Read contact high-school simulated dataset and retrun list of hyperedges
    along with timestamps and total no. of nodes.

    Returns:
    hyperedges: List. List of all hyperedges
    hyperedges_timestamps: List. Timestamp associated with each hyperedge.
        Timestamps should be in same order as that of hyperedges.
    total_nodes: Int. Total number of nodes in hypergraph.
    '''
    f = open('./data/chsim_groups.txt')
    hyperedges = [frozenset(map(int, line.strip().split(','))) for line in  f.readlines()]
    f.close()

    f = open('./data/chsim_node_information.txt')
    node_active = {int(line.strip().split()[0]): True if line.strip().split()[1] == 'Active' else False for line in f.readlines()}
    f.close()

    hyperedges = [hedge for hedge in hyperedges if all(
        node_active[node] for node in hedge)]
    return hyperedges, len(node_active)
