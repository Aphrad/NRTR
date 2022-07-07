import argparse
import math
import numpy
import numba
def get_args_parser():
    parser = argparse.ArgumentParser('NRTR_Neuron_Reconstruction_Transformer.pytorch', add_help=False)
    parser.add_argument('--in_swc', default="None", type=str)
    parser.add_argument('--out_swc', default="None", type=str)
    return parser


@numba.njit
def prim(graph, start=0):
    N = len(graph)
    k = start
    MST = dict()
    vis = numpy.zeros(N)
    vis[0] = 1

    while k < N-1:
        minw = numpy.inf
        u, v=0, 0
        for i in range(N):
            for j in range(N):
                if vis[i] ==1 and vis[j] == 0:
                    if graph[i,j] < minw:
                        minw = graph[i,j]
                        u,v=i,j
        vis[v] = 1
        k = k+1
        MST[v] = u

    return MST

def generateConnectivity(args):
    nodes = dict()
    in_swc = open(args.in_swc, "r")
    for node_message in in_swc:
        if node_message[0] == '#':
            continue
        id, type, z, y, x, r, pa = node_message.split()
        z, y, x, r = round(float(z)), round(float(y)), round(float(x)), float(r)
        key = "{:d}+{:d}+{:d}".format(x, y, z)
        value = r
        if key not in nodes.keys():
            nodes[key] = value
        else:
            if value > nodes[key]:
                nodes[key] = value
    graph = numpy.zeros(shape=(len(nodes), len(nodes)))
    for idx1 in range(len(nodes.keys())):
        print("{}/{}".format(idx1, len(nodes)))
        key1 = list(nodes.keys())[idx1]
        x1, y1, z1 = key1.split("+")
        x1, y1, z1 = round(float(x1)), round(float(y1)), round(float(z1))
        for idx2 in range(len(nodes.keys())):
            key2 = list(nodes.keys())[idx2]
            x2, y2, z2 = key2.split("+")
            x2, y2, z2 = round(float(x2)), round(float(y2)), round(float(z2))
            graph[idx1, idx2] = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2))
    # graph = numpy.loadtxt("graph.txt")
    path = prim(graph=graph, start=0)
    out_swc = open(args.out_swc, "w")
    for idx in range(len(nodes.keys())):
        key = list(nodes.keys())[idx]
        x, y, z = key.split("+")
        x, y, z = round(float(x)), round(float(y)), round(float(z))
        r = nodes[key]
        if idx in path.keys():
            pa = path[idx]
        else:
            pa = -1
        node_message = "{:d} {:d} {:d} {:d} {:d} {:.2f} {:d}\n".format(idx, 2, z, y, x, r, pa)
        out_swc.write(node_message)
    in_swc.close()
    out_swc.close()
    

if __name__=="__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    generateConnectivity(args)