from yafs.selection import Selection
from yafs.distribution import deterministicDistribution
from yafs.population import *
import networkx as nx
from yafs.topology import Topology
import random
import numpy as np
import time
from yafs.utils import fractional_selectivity
from yafs.stats import Stats
from simplePlacement import CloudPlacement
from yafs.topology import Topology
from yafs.placement import *
from yafs.population import *
from yafs.application import Application, Message
import argparse
import sys
import pandas as pd


def create_application():
    # APPLICATION
    a = Application(name="SimpleCase")

    # (S) --> (ServiceA) --> (A)
    a.set_modules([{"Sensor": {"Type": Application.TYPE_SOURCE}},
                   {"ServiceA": {"Type": Application.TYPE_MODULE}},
                   {"Actuator": {"Type": Application.TYPE_SINK}}
                   ])
    """
    Messages among MODULES (AppEdge in iFogSim)
    """
    m_a = Message("M.A", "Sensor", "ServiceA",
                  instructions=20*10 ^ 6, bytes=1000)
    m_b = Message("M.B", "ServiceA", "Actuator",
                  instructions=10*10 ^ 6, bytes=750)

    """
    Defining which messages will be dynamically generated # the generation is controlled by Population algorithm
    """
    a.add_source_messages(m_a)

    """
    MODULES/SERVICES: Definition of Generators and Consumers (AppEdges and TupleMappings in iFogSim)
    """
    # MODULE SERVICES
    a.add_service_module("ServiceA", m_a, m_b,
                         fractional_selectivity, threshold=1.0)

    return a


class CustomPath(Selection):

    def __init__(self, s):

        Selection.__init__(self, s)
        self.topology = Topology()
        self.number_of_compute_nodes = 0
        self.number_of_sensor_nodes = 0
        self.pop = None
        self.app = None

    def init_state(self, sim):
        self.states = []

        # self.latencies = []

        self.test_sensor = 0
        for node, val in sim.topology.nodeAttributes.items():
            if "sensor" in val["model"]:
                self.test_sensor = node
                break

    def set_population(self):

        self.app = create_application()

        self.pop = Statical("Statical")

        self.pop.set_sink_control(
            {"model": "actuator-device", "number": 1, "module": self.app.get_sink_modules()})

        dDistribution = deterministicDistribution(
            name="Deterministic", time=10)

        for i in range(self.number_of_sensor_nodes):

            self.pop.set_src_control({"model": "sensor-device-"+str(i+1), "number": 1,
                                      "message": self.app.get_message("M.A"), "distribution": dDistribution})

    def create_dynamic_links(self, t):

        for i in range(self.number_of_sensor_nodes):

            for j in range(self.number_of_compute_nodes):

                link = {"s": i, "d": j + self.number_of_sensor_nodes + 1, "BW": 1,
                                "PR": random.randint(1, 10)}

                t["link"].append(link)

        for j in range(self.number_of_compute_nodes):

            link = {"s": j + self.number_of_sensor_nodes + 1, "d": self.number_of_sensor_nodes, "BW": 1,
                    "PR": random.randint(1, 10)}

            t["link"].append(link)

        return t

    def create_dynamic_json_topology(self):
        """
        TOPOLOGY DEFINITION

        Some attributes of fog entities (nodes) are approximate

        """

        # MANDATORY FIELDS

        self.number_of_sensor_nodes = random.randint(4, 8)
        # print(self.number_of_sensor_nodes)
        self.number_of_compute_nodes = random.randint(1, 3)

        topology_json = {}
        topology_json["entity"] = []
        topology_json["link"] = []

        cloud_dev = {"mytag": "cloud",
                     "IPT": 2500 * 10 ^ 6, "RAM": 20000, "COST": 2, "WATT": 10.0, "devices": [], "device_bandwidth": 10, "services": set(), "unitilised_bandwidth": 0, "sensors_accessing": set(), "peak_memory": 0, "residual_memory": 20000}
        # change the number of sensors at an interval of 60 time units

        for i in range(self.number_of_sensor_nodes):

            sensor_dev = {"id": i, "model": "sensor-device-"+str(i+1),
                          "IPT": 100 * 10 ^ 6, "RAM": 4000, "COST": 3, "WATT": 40.0, "device_bandwidth": 500000}

            topology_json["entity"].append(sensor_dev)

        actuator_dev = {"id": self.number_of_sensor_nodes, "model": "actuator-device",
                        "IPT": 100 * 10 ^ 6, "RAM": 4000, "COST": 3, "WATT": 40.0}

        topology_json["entity"].append(actuator_dev)

        ID = self.number_of_sensor_nodes + 1

        for i in range(self.number_of_compute_nodes):

            cloud_dev["id"] = ID

            cloud_dev["model"] = "cloud-" + str(ID - 20)

            topology_json["entity"].append(cloud_dev.copy())

            ID += 1

        return topology_json

    def create_topology(self):

        t_json = self.create_dynamic_json_topology()

        t_json_final = self.create_dynamic_links(t_json)

        self.topology.load(t_json_final)

        self.topology.show()

    def update_topology(self, sim, app_name):

        value = {"mytag": "cloud"}  # or whatever tag
        id_cluster = self.topology.find_IDs(value)
        num_services = {}
        for one_node in id_cluster:
            # print "lol"
            # print sim.topology.nodeAttributes[one_node]["services"]
            num_services[one_node] = len(
                self.topology.nodeAttributes[one_node]["services"])
        print num_services

    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic, from_des):
        """
            Computes the minimun path among the source elemento of the topology and the localizations of the module

            Return the path and the identifier of the module deployed in the last element of that path
            """

        if self.var == 1:
            # print sim.topology.get_nodes_att()
            self.node_dict = sim.topology.get_nodes_att()
            print(self.node_dict)
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

            if node_src == self.test_sensor:
                print("This is the test sensor")
                # State Computation
                value = {"mytag": "cloud"}
                id_cluster = sim.topology.find_IDs(value)
                print("Updating.....")
                print(sim.env.now)

                current_state = dict()
                current_bandwidths = dict()
                current_prs = dict()
                cpu_utils = dict()
                memories = dict()
                all_links = sim.topology.get_edges()

                for edge_node in id_cluster:
                    one_link = (self.test_sensor, edge_node)
                    if one_link not in all_links:
                        one_link = (edge_node, self.test_sensor)
                    if one_link not in all_links:
                        continue
                    band_val = sim.topology.get_edge(
                        one_link)[sim.topology.LINK_BW]
                    pr_val = sim.topology.get_edge(one_link)['PR']
                    current_bandwidths[edge_node] = (band_val, band_val)
                    current_prs[edge_node] = pr_val
                    memories[edge_node] = (sim.topology.nodeAttributes[edge_node]["peak_memory"],
                                           sim.topology.nodeAttributes[edge_node]["residual_memory"])
                current_state["bandwidth"] = current_bandwidths
                current_state["PR"] = current_prs
                current_state["memories"] = memories
                current_state["input_size"] = message.bytes
                self.states.append(current_state)

                # change 5 to a global variable =  history of the states

                if len(self.states) > 5:
                    self.states.pop(0)

                print("printing states :")
                print(self.states)
                #####
            # Select Node
            destination_nodes = set()
            for a in DES_dst:
                destination_nodes.add(alloc_DES[a])
            print("destination nodes : ", destination_nodes)

            min_val = 1000000000
            global final_node
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
        # print("band : ")
        # print( sim.topology.nodeAttributes[final_node]["device_bandwidth"] )
        # print( "start node :")
        # print node_src
        # print( "final node :")
        # print final_node

            # update dict
        # print("lol")
            sim.topology.nodeAttributes[final_node]["sensors_accessing"].add(
                node_src)
            sim.update_bands()
            print("sensors accessing : ")
            print(
                sim.topology.nodeAttributes[final_node]["sensors_accessing"])

            size_bits = message.bytes

            link = (node_src, final_node)

            transmit = size_bits / (sim.topology.get_edge(link)
                                    [Topology.LINK_BW] * 1000000.0)

            propagation = sim.topology.get_edge(link)[Topology.LINK_PR]
            print " link bw"
            print sim.topology.get_edge(link)[Topology.LINK_BW]
            print "transmit"
            print transmit
            print "propagation"
            print propagation

            ipt = 1

            for key, value in self.node_dict[final_node].items():

                if key == 'IPT':

                    ipt = float(value)

            print "ipt"
            print ipt

            # self.dict[final_node] = sim.env.now
            # print("value is : \n", self.dict[final_node])
            # self.dict[final_node] = (message.inst / ipt) + transmit + propagation
            # print("value is : \n", self.dict[final_node])
            if self.dict[final_node] < sim.env.now + transmit + propagation:

                print("true")

                self.dict[final_node] = sim.env.now

                self.dict[final_node] += (message.inst / ipt) + \
                    transmit + propagation

            else:

                print("false")

                self.dict[final_node] += message.inst / ipt

            # print("new traffic : ", traffic)
            # print("updated dict : ", self.dict)
            # print("final node")
            # print(final_node)
            # print("final des")
            # print(final_des)
            # print message.inst
            # print self.node_dict[final_node]

        else:

            dst_node = 0
            for des in DES_dst:  # In this case, there are only one deployment
                dst_node = alloc_DES[des]
                print ("\t\t Looking the path to id_node: %i" % dst_node)

                path = list(nx.shortest_path(sim.topology.G,
                                             source=node_src, target=dst_node))

                bestPath = [path]
                bestDES = [des]

            print " link bw"
            print sim.topology.get_edge((node_src, dst_node))[
                Topology.LINK_BW]

        return bestPath, bestDES
