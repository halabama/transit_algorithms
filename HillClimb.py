import sys
import numpy
import time
import random
import math

from numpy.core.multiarray import ndarray
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path


def create_topology(n_nodes, dx, dy):
    node_list = []
    for i in range(0, n_nodes):
        node_list.append([random.randint(0, dx), random.randint(0, dy)])

    graph_data = numpy.empty((n_nodes, n_nodes))
    graph_data.fill(0)
    for i in range(0, n_nodes):
        for j in range(0, n_nodes):
            graph_data[i][j] = math.ceil(math.sqrt((node_list[i][0]-node_list[j][0])**2 +
                                                   (node_list[i][1]-node_list[j][1])**2))

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


def dijkstra(graph, source, sink):
    start = source
    finish = sink
    res, pre = shortest_path(graph, 'D', return_predecessors=True, indices=start)
    path = [finish]
    i = finish
    while i != start:
        i = pre[i]
        path.append(i)
    path.reverse()
    return path


def yen_ksp(graph, source, sink, K):
    a = [dijkstra(graph, source, sink)]
    b = []
    graph_copy = lil_matrix(graph.copy())
    for k in range(1, K+1):
        for i in range(0, len(a[k-1])-1):

            spur_node = a[k-1][i]
            root_path = a[k-1][0:i]
            for path in a:
                if root_path == path[0:i]:
                    graph_copy[a[k-1][i], a[k-1][i+1]] = float('inf')
                    graph_copy[a[k - 1][i+1], a[k - 1][i]] = float('inf')
            for node in root_path:
                if node != spur_node:
                    graph_copy[node, :] = float('inf')
                    graph_copy[:, node] = float('inf')

            spur_path = dijkstra(graph_copy, spur_node, sink)
            root_path.extend(spur_path)
            total_path = root_path
            b.append(total_path)

            graph_copy = lil_matrix(graph.copy())

        if not b:
            break

        b_cost = cost(graph, b)
        minim = numpy.argmin(b_cost)
        a.append(b[minim])
        b.pop(minim)
    return a


def cost(graph, paths):
    c = []
    for i in range(0, len(paths)):
        c.append(0)
        for n in range(1, len(paths[i])):
            c[i] += graph[paths[i][n - 1], paths[i][n]]
    return c


def create_path_from_prev(prev):
    paths = [[[]]]
    for i in range(0, len(prev[0])):
        paths.append([[]])
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
                paths[i][j].extend(path)
    return paths


def initialize_route_generation(graph, demand: numpy.array, num_routes, min_node, max_node):
    dist, prev = shortest_path(graph, 'D', return_predecessors=True)
    edges = numpy.array(graph.copy())
    edges.fill(0)
    sp_list = create_path_from_prev(prev)
    for i in range(0, len(graph[0])):
        for j in range(0, len(graph[0])):
            for k in range(1, len(sp_list[i][j])):
                edges[sp_list[i][j][k-1]][sp_list[i][j][k]] += demand[i][j]

    for i in range(0, len(graph[0])):
        for j in range(0, len(graph[0])):
            if i != j and graph[i][j] != float('inf'):
                edges[i][j] /= graph[i][j]
            else:
                edges[i][j] = -float('inf')

    active_edges = []
    for i in range(0, len(edges[0])):
        for j in range(i, len(edges[0])):
            edge_sum = edges[i][j]+edges[j][i]
            edges[i][j] = edge_sum
            edges[j][i] = edge_sum
            if edges[i][j] != -float('inf'):
                active_edges.append([i, j])
                active_edges.append([j, i])

    solution_list = []
    for i in range(0, num_routes):
        random_edge = random.randint(0, len(active_edges)-1)
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


def evaluate_solution(graph: numpy.array, demand, solution):
    score = 0
    routes_graph = numpy.empty((len(solution), len(graph[0]), len(graph[0])))
    routes_dist = numpy.empty((len(solution), len(graph[0]), len(graph[0])))

    routes_graph.fill(float('inf'))
    for r in range(0, len(solution)):
        route = solution[r]
        for i in range(1, len(route)):
            routes_graph[r][route[i-1]][route[i]] = graph[route[i-1]][route[i]]
            routes_graph[r][route[i]][route[i-1]] = graph[route[i]][route[i-1]]
        routes_dist[r] = shortest_path(routes_graph[r], 'D')

    total_graph = numpy.empty((len(graph[0]), len(graph[0])))
    total_graph.fill(float('inf'))
    for route in routes_graph:
        total_graph = numpy.minimum(route, total_graph)
    total_dist = shortest_path(total_graph, 'D')

    for i in range(0, len(graph[0])):
        for j in range(0, len(graph[0])):
            min_value = numpy.min(routes_dist, 0)[i][j]
            if min_value != float('inf'):
                score += demand[i][j] * min_value
            else:
                if demand[i][j] != 0:
                    score += demand[i][j] * (total_dist[i][j]*1.25)
    return score


def solution_modification(graph, demand, max_node, min_node, solution):
    random_route = random.randint(0, len(solution)-1)
    solution_route = list(solution[random_route].copy())
    modified_solution = solution.copy()
    rand_node = 0

    for node in range(0, len(solution_route)):
        if len(solution_route) <= min_node:
            break
        if node != 0 and node != len(solution_route)-1:
            if graph[solution_route[node-1]][solution_route[node+1]] == float('inf'):
                continue

        new_route = numpy.delete(solution_route, node)
        new_solution = solution.copy()
        new_solution[random_route] = new_route

        if evaluate_solution(graph, demand, new_solution) <= evaluate_solution(graph, demand, modified_solution):
            modified_solution = new_solution

    if len(solution_route) >= max_node:
        solution_route.pop(0)
    while rand_node in solution_route:
        rand_node = random.randint(0, len(graph[0])-1)

    for node in range(0, len(solution_route)+1):
        if node == 0:
            if graph[rand_node][solution_route[node]] == float('inf'):
                continue
        elif node == len(solution_route):
            if graph[solution_route[node-1]][rand_node] == float('inf'):
                continue
        else:
            if graph[solution_route[node-1]][rand_node] == float('inf') or\
                    graph[rand_node][solution_route[node]] == float('inf'):
                continue

        new_route = numpy.insert(solution_route, node, rand_node)
        new_solution = solution.copy()
        new_solution[random_route] = new_route

        score_new = evaluate_solution(graph, demand, new_solution)
        score_mod = evaluate_solution(graph, demand, modified_solution)
        if score_new < score_mod or score_mod == float('inf'):
            modified_solution = new_solution

    return modified_solution


def route_generation(graph, demand, num_routes, min_node, max_node, target_score):
    time.clock()
    css = initialize_route_generation(graph, demand, num_routes, min_node, max_node)
    sc = evaluate_solution(graph, demand, css)
    nss = css.copy()
    while sc > target_score and time.clock() < 10:
        nss = solution_modification(graph, demand, max_node, min_node, nss)
        sn = evaluate_solution(graph, demand, nss)
        if sn <= sc:
            css = nss
            sc = sn
        else:
            nss = css

    return sc, css


class Vehicle:
    capacity = 0
    velocity = 0

    def __init__(self, p_capacity, p_velocity):
        self.capacity = p_capacity
        self.velocity = p_velocity


def frequency_setting(graph, demand, solution, vehicle: Vehicle):
    routes_graph = numpy.empty((len(solution), len(graph[0]), len(graph[0])))
    routes_dist = numpy.empty((len(solution), len(graph[0]), len(graph[0])))
    routes_edge_load = numpy.empty((len(solution), len(graph[0]), len(graph[0])))
    routes_paths = []
    dynamic_demand = demand.copy()
    routes_edge_load.fill(0)
    routes_graph.fill(float('inf'))
    for r in range(0, len(solution)):
        route = solution[r]
        for i in range(1, len(route)):
            routes_graph[r][route[i - 1]][route[i]] = graph[route[i - 1]][route[i]]
            routes_graph[r][route[i]][route[i - 1]] = graph[route[i]][route[i - 1]]
        routes_dist[r], routes_prev = shortest_path(routes_graph[r], 'D', return_predecessors=True)
        routes_paths.append(create_path_from_prev(routes_prev))

    min_value = numpy.min(routes_dist, 0)
    for i in range(0, len(graph[0])):
        for j in range(0, len(graph[0])):
            if i == j:
                continue
            if min_value[i][j] != float('inf'):
                routes_d = []
                for r in range(0, len(solution)):
                    if routes_dist[r][i][j] == min_value[i][j]:
                        routes_d.append(r)

                routes = []
                for r in routes_d:
                    if not routes:
                        routes.append(r)
                        continue
                    tmp = routes[0]
                    if routes_dist[r][solution[r][0]][solution[r][-1]] < \
                            routes_dist[tmp][solution[tmp][0]][solution[tmp][-1]]:
                        routes.clear()
                        routes.append(r)
                    elif routes_dist[r][solution[r][0]][solution[r][-1]] == \
                            routes_dist[tmp][solution[tmp][0]][solution[tmp][-1]]:
                        routes.append(r)

                r_demand = dynamic_demand[i][j]/len(routes)
                dynamic_demand[i][j] = 0
                for route in routes:
                    path = routes_paths[route][i][j]
                    for x in range(1, len(path)):
                        routes_edge_load[route][path[x-1]][path[x]] += r_demand

    if dynamic_demand.sum() > 0:
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

        min_value = numpy.min(transfer_dist, axis=(0, 1))

        for i in range(0, len(demand)):
            for j in range(0, len(demand)):
                if i == j or dynamic_demand[i][j] == 0 or min_value[i][j] == float('inf'):
                    continue
                route_pair_d = []
                for rx in range(0, len(solution)):
                    for ry in range(rx+1, len(solution)):
                        if transfer_dist[rx][ry][i][j] == min_value[i][j]:
                            route_pair_d.append([rx, ry])

                route_pair = []
                for pair in route_pair_d:
                    if not route_pair:
                        route_pair.append(pair)
                        continue
                    cycle_sum_new = routes_dist[pair[0]][solution[pair[0]][0]][solution[pair[0]][-1]] +\
                        routes_dist[pair[1]][solution[pair[1]][0]][solution[pair[1]][-1]]
                    cycle_sum_old = \
                        routes_dist[route_pair[0][0]][solution[route_pair[0][0]][0]][solution[route_pair[0][0]][-1]] +\
                        routes_dist[route_pair[0][1]][solution[route_pair[0][1]][0]][solution[route_pair[0][1]][-1]]
                    if cycle_sum_new < cycle_sum_old:
                        route_pair.clear()
                        route_pair.append(pair)
                    elif cycle_sum_old == cycle_sum_new:
                        route_pair.append(pair)

                r_demand = dynamic_demand[i][j]/len(route_pair)
                dynamic_demand[i][j] = 0
                for pair in route_pair:
                    route_x = list(solution[pair[0]])
                    route_y = list(solution[pair[1]])
                    path = transfer_paths[pair[0]][pair[1]-pair[0]-1][i][j]
                    for p in range(1, len(path)):
                        if path[p-1] in route_x and path[p] in route_x:
                            if abs(route_x.index(path[p-1]) - route_x.index(path[p])) == 1:
                                routes_edge_load[pair[0]][path[p-1]][path[p]] += r_demand
                        elif path[p-1] in route_y and path[p] in route_y:
                            if abs(route_y.index(path[p-1]) - route_y.index(path[p])) == 1:
                                routes_edge_load[pair[1]][path[p-1]][path[p]] += r_demand

    frequency_set = numpy.empty(len(solution))
    operation_set = numpy.empty(len(solution))
    for i in range(0, len(solution)):
        frequency_set[i] = (numpy.max(routes_edge_load[i]) / vehicle.capacity)
        operation_set[i] = math.ceil(2*(routes_dist[i][solution[i][0]][solution[i][-1]]/vehicle.velocity)
                                     * (frequency_set[i]/3600))
    return frequency_set, operation_set


def main():
    # args = sys.argv[1:]
    # graph_data = numpy.genfromtxt(args[0], delimiter=";")
    graph_data = create_topology(25, 1000, 1000)
    # demand_per_hour: ndarray = numpy.genfromtxt(args[1], delimiter=";")
    demand_per_hour = create_demand_matrix(graph_data, 100)
    sc, solution = route_generation(graph_data, demand_per_hour, random.randint(1, len(graph_data[0])),
                                    2, len(graph_data[0]), 1201)
    vehicle = Vehicle(20, 14)
    freq_set, opr_score_set = frequency_setting(graph_data, demand_per_hour, solution, vehicle)
    print("Usage Score:%.2f" % sc, " Operation Score:%d" % opr_score_set.sum())
    for i in range(0, len(solution)):
        print("Route ", i, ",freq_per_hour=%.2f,vehicles=%d" % (freq_set[i], opr_score_set[i]), "\t:\t",
              '-'.join(map(str, solution[i])), sep="")


if __name__ == "__main__":
    main()
