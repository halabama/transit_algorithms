import numpy
import random
import math
import sys
from scipy.sparse.csgraph import shortest_path


def create_topology(n_nodes, dx, dy):
    node_list = []
    for i in range(0, n_nodes):
        node_list.append([random.randint(0, dx), random.randint(0, dy)])

    graph_data = numpy.empty((n_nodes, n_nodes))
    graph_data.fill(float('inf'))
    for i in range(0, n_nodes):
        for j in range(0, n_nodes):
            if i == j:
                continue
            graph_data[i][j] = math.ceil(math.sqrt((node_list[i][0]-node_list[j][0])**2 +
                                                   (node_list[i][1]-node_list[j][1])**2))
            if random.random() < 0.25:
                graph_data[i][j] = float('inf')
    return graph_data


def create_demand_matrix(graph_data, max_node_population):
    traffic_count = []
    for i in range(0, len(graph_data[0])):
        traffic_count.append(random.randint(1, max_node_population))

    demand_data = numpy.empty((len(graph_data[0]), len(graph_data[0])))
    demand_data.fill(0)
    for i in range(0, len(demand_data[0])):
        tmp_sum = numpy.sum(traffic_count) - traffic_count[i]
        for j in range(i+1, len(demand_data[0])):
            demand_data[i][j] = math.ceil(traffic_count[i] * traffic_count[j]/tmp_sum)
    demand_data = numpy.add(demand_data, demand_data.transpose())
    return demand_data


def create_path_from_prev(prev):
    paths = []
    for i in range(0, len(prev[0])):
        paths.append([])
        for j in range(0, len(prev[0])):
            paths[i].append([])
            if i != j:
                start = i
                finish = j
                path = [finish]
                while start != finish:
                    if finish < 0:
                        path.clear()
                        break
                    finish = prev[i][finish]
                    path.append(finish)
                path.reverse()
                if path:
                    if path[0] == i and path[-1] == j:
                        paths[i][j].append(path)
                    else:
                        paths[i][j].append([])
                else:
                    paths[i][j].append([])
    return paths


def yen_ksp(graph, k):

    dist, prev = shortest_path(graph, 'D', return_predecessors=True)
    res = create_path_from_prev(prev)
    graph_copy = graph.copy()

    for i in range(0, len(graph)):
        for j in range(0, len(graph)):
            if i == j:
                continue
            b = []
            for r in range(1, k):
                for x in range(0, len(res[i][j][-1])-1):
                    spur_node = res[i][j][-1][x]
                    root_path = res[i][j][-1][0:x+1]

                    for path in res[i][j]:
                        if root_path == path[0:x+1]:
                            graph_copy[path[x]][path[x+1]] = float('inf')

                    for root_node in root_path:
                        if root_node != spur_node:
                            graph_copy[root_node][:] = float('inf')
                            graph_copy[:][root_node] = float('inf')
                    dist, prev = shortest_path(graph_copy, 'D', return_predecessors=True)
                    spur_path = create_path_from_prev(prev)[spur_node][j][0]
                    total_path = []
                    if spur_path:
                        if root_path:
                            root_path.pop(-1)
                        total_path = root_path + spur_path
                    cost = 0
                    for n in range(1, len(total_path)):
                        cost += graph[total_path[n-1]][total_path[n]]
                    if cost != 0 and cost != float('inf'):
                        b.append([n, total_path])
                    graph_copy = graph.copy()
                if not b:
                    break
                b.sort()
                if b[0][1] not in res[i][j]:
                    res[i][j].append(b[0][1])
                b.pop(0)
    return res


def icrsgp(graph, min_node, max_node, k):
    candidates = []
    tmp = yen_ksp(graph, k)
    for i in range(0, len(graph)):
        for j in range(0, len(graph)):
            for r in range(0, len(tmp[i][j])):
                if min_node <= len(tmp[i][j][r]) <= max_node:
                    candidates.append(tmp[i][j][r])
    return candidates


def init_hill_climbing(graph, demand, num_routes, min_node, max_node):
    dist, prev = shortest_path(graph, 'D', return_predecessors=True)
    edges = numpy.array(graph.copy())
    edges.fill(0)
    sp_list = create_path_from_prev(prev)
    for i in range(0, len(graph[0])):
        for j in range(0, len(graph[0])):
            for k in range(1, len(sp_list[i][j])):
                edges[sp_list[i][j][k - 1]][sp_list[i][j][k]] += demand[i][j]

    for i in range(0, len(graph[0])):
        for j in range(0, len(graph[0])):
            if i != j and graph[i][j] != float('inf'):
                edges[i][j] /= graph[i][j]
            else:
                edges[i][j] = -float('inf')

    active_edges = []
    for i in range(0, len(edges[0])):
        for j in range(i, len(edges[0])):
            edge_sum = edges[i][j] + edges[j][i]
            edges[i][j] = edge_sum
            edges[j][i] = edge_sum
            if edges[i][j] != -float('inf'):
                active_edges.append([i, j])
                active_edges.append([j, i])

    solution_list = []
    for i in range(0, num_routes):
        random_edge = random.randint(0, len(active_edges) - 1)
        new_route = [active_edges[random_edge][0], active_edges[random_edge][1]]
        num_nodes = random.randint(min_node, max_node)
        tmp_edges = numpy.array(edges)
        tmp_edges[:, new_route[-1]] = -float('inf')
        tmp_edges[new_route[0], :] = -float('inf')
        tmp_edges[new_route[-1], new_route[0]] = -float('inf')
        for n in range(2, num_nodes):
            max_incoming = tmp_edges.argmax(axis=0)
            max_outgoing = tmp_edges.argmax(axis=1)
            extend_end_candidate = max_outgoing[new_route[-1]]
            extend_start_candidate = max_incoming[new_route[0]]
            if tmp_edges[new_route[-1]][extend_end_candidate] > tmp_edges[extend_start_candidate][new_route[0]]:
                tmp_edges[new_route[-1], :] = -float('inf')
                new_route.append(extend_end_candidate)
                tmp_edges[:, new_route[-1]] = -float('inf')
            elif tmp_edges[extend_start_candidate][new_route[0]] > -float('inf'):
                tmp_edges[:, new_route[0]] = -float('inf')
                new_route.insert(0, extend_start_candidate)
                tmp_edges[new_route[0], :] = -float('inf')
            tmp_edges[new_route[-1]][new_route[0]] = -float('inf')
        solution_list.append(new_route)
    return solution_list


class Vehicle:
    capacity = 0
    velocity = 0

    def __init__(self, p_capacity, p_velocity):
        self.capacity = p_capacity
        self.velocity = p_velocity


def calculate_solution_stats(graph, solution):
    routes_graph = numpy.empty((len(solution), len(graph[0]), len(graph[0])))
    routes_dist = numpy.empty((len(solution), len(graph[0]), len(graph[0])))
    routes_paths = []
    routes_length = numpy.empty((len(solution)))
    routes_graph.fill(float('inf'))
    for r in range(0, len(solution)):
        route = solution[r]
        for i in range(1, len(route)):
            routes_graph[r][route[i - 1]][route[i]] = graph[route[i - 1]][route[i]]
            routes_graph[r][route[i]][route[i - 1]] = graph[route[i]][route[i - 1]]
        routes_dist[r], routes_prev = shortest_path(routes_graph[r], 'D', return_predecessors=True)
        routes_paths.append(create_path_from_prev(routes_prev))
        routes_length[r] = routes_dist[r][route[0]][route[-1]]

    transfer_graph = numpy.empty((len(solution), len(solution), len(graph[0]), len(graph[0])))
    transfer_dist = numpy.empty((len(solution), len(solution), len(graph[0]), len(graph[0])))
    transfer_paths = []
    transfer_graph.fill(float('inf'))
    transfer_dist.fill(float('inf'))
    for rx in range(0, len(solution)):
        transfer_paths.append([])
        for ry in range(rx, len(solution)):
            if rx != ry:
                transfer_graph[rx][ry] = numpy.minimum(routes_graph[rx], routes_graph[ry])
                transfer_dist[rx][ry], t_prev = shortest_path(transfer_graph[rx][ry], 'D', return_predecessors=True)
                transfer_paths[rx].append(create_path_from_prev(t_prev))
    return routes_graph, routes_dist, routes_paths, routes_length, transfer_dist, transfer_paths


def populate_routes(routes_dist, routes_paths, routes_length, transfer_dist, transfer_paths, demand, solution):

    routes_edge_load = numpy.empty((len(routes_dist), len(demand), len(demand)))
    dynamic_demand = demand.copy()
    routes_edge_load.fill(0)

    min_value = numpy.min(routes_dist, 0)
    for i in range(0, len(demand)):
        for j in range(0, len(demand)):
            if i == j:
                continue
            if min_value[i][j] != float('inf'):
                routes = numpy.nonzero(routes_dist[:, i, j] == min_value[i][j])[0]
                od_routes_length = numpy.array([routes_length[r] for r in routes])
                tot_len = od_routes_length.sum()
                len_scale = tot_len * (len(od_routes_length)-1)
                for r_id in range(0, len(routes)):
                    path = routes_paths[routes[r_id]][i][j][0]
                    if len_scale == 0:
                        r_demand = demand[i][j]
                    else:
                        r_demand = demand[i][j] * (tot_len-od_routes_length[r_id])/len_scale
                    for x in range(1, len(path)):
                        routes_edge_load[routes[r_id]][path[x - 1]][path[x]] += r_demand
                dynamic_demand[i][j] = 0

    if dynamic_demand.sum() > 0:

        min_value = numpy.min(transfer_dist, axis=(0, 1))
        for i in range(0, len(demand)):
            for j in range(0, len(demand)):
                if i == j or dynamic_demand[i][j] == 0 or min_value[i][j] == float('inf'):
                    continue
                route_pairs = numpy.array(numpy.nonzero(transfer_dist[:, :, i, j] == min_value[i][j])).transpose()
                od_pair_length = numpy.array([routes_length[pair[0]] + routes_length[pair[1]] for pair in route_pairs])
                tot_len = od_pair_length.sum()
                len_scale = tot_len * (len(od_pair_length) - 1)

                dynamic_demand[i][j] = 0
                for pair_id in range(0, len(route_pairs)):
                    pair = route_pairs[pair_id]
                    route_x = list(solution[pair[0]])
                    route_y = list(solution[pair[1]])
                    if len_scale == 0:
                        pair_demand = demand[i][j]
                    else:
                        pair_demand = demand[i][j] * (tot_len - od_pair_length[pair_id]) / len_scale
                    path = transfer_paths[pair[0]][pair[1]-pair[0]-1][i][j][0]
                    for p in range(1, len(path)):
                        if path[p-1] in route_x and path[p] in route_x:
                            if abs(route_x.index(path[p-1]) - route_x.index(path[p])) == 1:
                                routes_edge_load[pair[0]][path[p-1]][path[p]] += pair_demand
                        elif path[p-1] in route_y and path[p] in route_y:
                            if abs(route_y.index(path[p-1]) - route_y.index(path[p])) == 1:
                                routes_edge_load[pair[1]][path[p-1]][path[p]] += pair_demand

    route_population = numpy.empty(len(solution))

    for i in range(0, len(solution)):
        route_population[i] = numpy.max(routes_edge_load[i])
    return route_population, dynamic_demand.sum()


def evaluate_user_score(routes_graph, routes_dist, demand):

    score = 0

    total_graph = numpy.empty((len(demand), len(demand)))
    total_graph.fill(float('inf'))
    for graph in routes_graph:
        total_graph = numpy.minimum(graph, total_graph)
    total_dist = shortest_path(total_graph, 'D')

    for i in range(0, len(demand)):
        for j in range(0, len(demand)):
            min_value = numpy.min(routes_dist[:, i, j])
            if min_value != float('inf'):
                score += demand[i][j] * min_value
            else:
                if demand[i][j] != 0:
                    score += demand[i][j] * (total_dist[i][j] * 1.25)
    return score


def evaluate_operation_score(routes_dist, routes_paths, routes_length,
                             transfer_dist, transfer_paths, demand, solution, vehicle: Vehicle):

    routes_load, remain_demand = \
        populate_routes(routes_dist, routes_paths, routes_length, transfer_dist, transfer_paths, demand, solution)
    routes_frequency = [r/vehicle.capacity for r in routes_load]

    operation_score = 0
    for i in range(0, len(routes_frequency)):
        operation_score += math.ceil(2*(routes_length[i]/vehicle.velocity) * (routes_frequency[i]/3600))
    return operation_score, remain_demand, routes_frequency


def evaluate_solution(graph, demand, solution, vehicle: Vehicle, weights):
    routes_graph, routes_dist, routes_paths, routes_length, transfer_dist, transfer_paths = \
        calculate_solution_stats(graph, solution)
    user_score = evaluate_user_score(routes_graph, routes_dist, demand)
    operation_score, remain_demand, freq_set = \
        evaluate_operation_score(routes_dist, routes_paths, routes_length,
                                 transfer_dist, transfer_paths, demand, solution, vehicle)
    return weights[0]*user_score+weights[1]*operation_score+weights[2]*remain_demand, freq_set


def create_neighbour(candidate_space, current_solution: list):
    mod_index = random.randint(0, len(current_solution)-1)
    pivot = random.randint(0, len(candidate_space)-1)
    sign = random.randint(0, 1)*2-1
    for i in range(0, len(candidate_space)):
        pivot = (pivot+sign) % len(candidate_space)
        if pivot not in current_solution:
            break
    neighbour_solution = current_solution.copy()
    neighbour_solution.pop(mod_index)
    neighbour_solution.insert(mod_index, pivot)
    return neighbour_solution


def simulated_annealing(graph, demand, weights=(1, 1, 1),
                        max_routes=5, max_generation=2, max_cooling=3, max_counter=150):
    candidate_space = icrsgp(graph, 5, len(demand), 3)
    candidate_space.extend(init_hill_climbing(graph, demand, (len(demand)**2)*3, 3, len(demand)))

    optimal_solution = None
    optimal_score = float('inf')
    optimal_freq_set = None
    vehicle = Vehicle(60, 0.9)
    progress = 0
    for n in range(1, max_routes+1):
        for g in range(0, max_generation):
            initial_solution = []
            while len(initial_solution) < n:
                r = random.randint(0, len(candidate_space)-1)
                if r in initial_solution:
                    continue
                initial_solution.append(r)
            current_solution = initial_solution
            current_score, current_freq_set = \
                evaluate_solution(graph, demand, [candidate_space[r] for r in current_solution], vehicle, weights)
            for t in range(0, max_cooling):
                for i in range(0, max_counter):
                    neighbour = create_neighbour(candidate_space, current_solution)
                    neighbour_score, neighbour_freq_set =\
                        evaluate_solution(graph, demand, [candidate_space[r] for r in neighbour], vehicle, weights)
                    if numpy.min(neighbour_freq_set) < 0.5:
                        continue
                    if neighbour_score <= current_score:
                        current_solution = neighbour
                        current_score = neighbour_score
                        current_freq_set = neighbour_freq_set
                    elif random.random() <= math.exp(-neighbour_score*(t+1)/current_score):
                            current_solution = neighbour
                            current_score = neighbour_score
                            current_freq_set = neighbour_freq_set
                progress = (((n - 1) / max_routes) + (g / (max_generation * max_routes)) +
                            (t / (max_cooling * max_generation * max_routes))) * 100
                print("\rProgress:%.2f%%" % progress, end="")
                sys.stdout.flush()
            if current_score <= optimal_score:
                optimal_solution = current_solution
                optimal_score = current_score
                optimal_freq_set = current_freq_set
    return [candidate_space[c] for c in optimal_solution], optimal_score, optimal_freq_set


def main():
    graph_data = create_topology(20, 1000, 1000)
    demand = create_demand_matrix(graph_data, 100)
    opt_solution, opt_score, opt_freq_set = simulated_annealing(graph_data, demand, weights=(1e-4, 1, 0))
    print("\rOptimal Score : %.2f" % opt_score, end="")
    if opt_score == float('inf'):
        graph_dist = shortest_path(graph_data, 'D')
        print("=> Possible value:", numpy.max(graph_dist), end="")
    print("")
    for i in range(0, len(opt_solution)):
        opt_headway = "inf"
        if opt_freq_set[i] != 0:
            opt_headway = f'{(60/opt_freq_set[i]):.2f}'
        print("\tRoute ", i, "|Headway ", opt_headway, " min", "\t:\t",
              f'{"-".join(map(str, opt_solution[i])):<{len(demand)*3}}', sep="")


if __name__ == "__main__":
    main()
