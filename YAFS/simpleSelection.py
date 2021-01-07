
from yafs.selection import Selection
import networkx as nx
from yafs.topology import Topology


class CustomPath(Selection):

    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic, from_des):
        """
        Computes the minimun path among the source elemento of the topology and the localizations of the module

        Return the path and the identifier of the module deployed in the last element of that path
        """
        if self.var == 1:
            print sim.topology.get_nodes_att()
            self.node_dict = sim.topology.get_nodes_att()
            for key, value in self.node_dict.items():
                if 'cloud' in value["model"]:
                    self.dict[key] = sim.env.now
            print self.dict
            self.var = 0

        node_src = topology_src
        DES_dst = alloc_module[app_name][message.dst]

        curr_traff = traffic
        print ("traffic : ", traffic)
        print ("GET PATH")
        # print(alloc_DES)
        # print(sim.topology.get_nodes_att())
        # print(sim.__get_id_process())
        # print(sim.get_stats())
        print ("\tNode _ src (id_topology): %i" % node_src)
        print ("\tRequest service: %s " % message.dst)
        print ("\tProcess serving that service: %s " % DES_dst)

        if message.src == "Sensor":
            # Select Node
            destination_nodes = set()
            for a in DES_dst:
                destination_nodes.add(alloc_DES[a])
            print("destination nodes : ", destination_nodes)

            min_val = 1000000000
            final_node = 0
            for single_node in destination_nodes:

                if self.dict[single_node] < min_val:
                    min_val = self.dict[single_node]
                    final_node = single_node

            path = list(nx.shortest_path(sim.topology.G,
                                         source=node_src, target=final_node))
            bestPath = [path]
            alloc_des_reverse = {v: k for k, v in alloc_DES.iteritems()}
            final_des = alloc_des_reverse[final_node]
            bestDES = [final_des]

            # update dict

            size_bits = message.bytes

            link = (node_src, final_node)

            transmit = size_bits / (sim.topology.get_edge(link)
                                    [Topology.LINK_BW] * 1000000.0)

            propagation = sim.topology.get_edge(link)[Topology.LINK_PR]

            ipt = 1

            for key, value in self.node_dict[final_node].items():

                if key == 'IPT':

                    ipt = float(value)

            self.dict[final_node] = sim.env.now
            print("value is : \n", self.dict[final_node])
            self.dict[final_node] = (
                message.inst / ipt) + transmit + propagation
            print("value is : \n", self.dict[final_node])
            # if self.dict[final_node] < sim.env.now + transmit + propagation:

            #     print("true")

            #     self.dict[final_node] = sim.env.now

            #     self.dict[final_node] += (message.inst / ipt) + \
            #         transmit + propagation

            # else:

            #     print("false")

            #     self.dict[final_node] += message.inst / ipt

            print("new traffic : ", traffic)
            print("updated dict : ", self.dict)
            # print("final node")
            # print(final_node)
            # print("final des")
            # print(final_des)
            # print message.inst
            # print self.node_dict[final_node]

        else:
            for des in DES_dst:  # In this case, there are only one deployment
                dst_node = alloc_DES[des]
                print ("\t\t Looking the path to id_node: %i" % dst_node)

                path = list(nx.shortest_path(sim.topology.G,
                                             source=node_src, target=dst_node))

                bestPath = [path]
                bestDES = [des]

        return bestPath, bestDES


class MinimunPath(Selection):

    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic, from_des):
        """
        Computes the minimun path among the source elemento of the topology and the localizations of the module

        Return the path and the identifier of the module deployed in the last element of that path
        """

        node_src = topology_src
        DES_dst = alloc_module[app_name][message.dst]

        print ("GET PATH")
        print ("\tNode _ src (id_topology): %i" % node_src)
        print ("\tRequest service: %s " % message.dst)
        print ("\tProcess serving that service: %s " % DES_dst)

        print "traffic is : ", traffic
        bestPath = []
        bestDES = []

        # best destination based on the availablity of resources

        for des in DES_dst:  # In this case, there are only one deployment
            dst_node = alloc_DES[des]
            print ("\t\t Looking the path to id_node: %i" % dst_node)
            print "node source : ", node_src
            print "destination node : ", dst_node

            # These lines have to be modified for the project
            path = list(nx.shortest_path(sim.topology.G,
                                         source=node_src, target=dst_node, weight="BW"))
            print "path : ", path
            bestPath = [path]
            bestDES = [des]

        return bestPath, bestDES


class MinPath_RoundRobin(Selection):

    def __init__(self):
        self.rr = {}  # for a each type of service, we have a mod-counter

    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic, from_des):
        """
        Computes the minimun path among the source elemento of the topology and the localizations of the module

        Return the path and the identifier of the module deployed in the last element of that path
        """
        node_src = topology_src
        # returns an array with all DES process serving
        DES_dst = alloc_module[app_name][message.dst]

        if message.dst not in self.rr.keys():
            self.rr[message.dst] = 0

        print ("GET PATH")
        print ("\tNode _ src (id_topology): %i" % node_src)
        print ("\tRequest service: %s " % (message.dst))
        print ("\tProcess serving that service: %s (pos ID: %i)" %
               (DES_dst, self.rr[message.dst]))

        print "traffic is : ", traffic

        # print os.path.isfile("Results_multiple.csv")
        # if os.path.isfile("Results_multiple.csv"):

        #     get_last_row_from_csv("Results_multiple.csv")

        bestPath = []
        bestDES = []

        for ix, des in enumerate(DES_dst):
            if message.name == "M.A":

                print self.rr
                if self.rr[message.dst] == ix:
                    dst_node = alloc_DES[des]

                    path = list(nx.shortest_path(sim.topology.G,
                                                 source=node_src, target=dst_node, weight='BW'))

                    bestPath = [path]
                    bestDES = [des]

                    self.rr[message.dst] = (
                        self.rr[message.dst] + 1) % len(DES_dst)
                    break
            else:  # message.name == "M.B"

                dst_node = alloc_DES[des]

                path = list(nx.shortest_path(sim.topology.G,
                                             source=node_src, target=dst_node))
                if message.broadcasting:
                    bestPath.append(path)
                    bestDES.append(des)
                else:
                    bestPath = [path]
                    bestDES = [des]

        print "best path is : ", bestPath

        print "the best Des is : ", bestDES

        return bestPath, bestDES
