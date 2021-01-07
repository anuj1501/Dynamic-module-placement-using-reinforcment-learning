"""

    Created on Wed Nov 22 15:03:21 2017

    @author: isaac

"""
import random

import argparse

from yafs.core import Sim

from yafs.application import Application, Message

from yafs.population import *

from yafs.topology import Topology

from simpleSelection import CustomPath

from simplePlacement import CloudPlacement

from yafs.stats import Stats

from yafs.distribution import deterministicDistribution

from yafs.utils import fractional_selectivity

import time
import numpy as np

RANDOM_SEED = 1


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


def create_json_topology():
    """
       TOPOLOGY DEFINITION

       Some attributes of fog entities (nodes) are approximate
       """

    # MANDATORY FIELDS
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []

    cloud_dev_1 = {"id": 0, "model": "cloud-1", "mytag": "cloud",
                   "IPT": 2500 * 10 ^ 6, "RAM": 20000, "COST": 2, "WATT": 10.0}
    cloud_dev_2 = {"id": 4, "model": "cloud-2", "mytag": "cloud",
                   "IPT": 1500 * 10 ^ 6, "RAM": 10000, "COST": 2, "WATT": 10.0}
    sensor_dev_1 = {"id": 1, "model": "sensor-device-1",
                    "IPT": 100 * 10 ^ 6, "RAM": 4000, "COST": 3, "WATT": 40.0}
    sensor_dev_2 = {"id": 3, "model": "sensor-device-2",
                    "IPT": 200 * 10 ^ 6, "RAM": 8000, "COST": 2, "WATT": 40.0}
    actuator_dev = {"id": 2, "model": "actuator-device",
                    "IPT": 100 * 10 ^ 6, "RAM": 4000, "COST": 3, "WATT": 40.0}

    link1 = {"s": 1, "d": 0, "BW": 1, "PR": 1}
    link2 = {"s": 3, "d": 0, "BW": 1, "PR": 1}

    link3 = {"s": 1, "d": 4, "BW": 1, "PR": 1}
    link4 = {"s": 3, "d": 4, "BW": 1, "PR": 1}

    link5 = {"s": 0, "d": 2, "BW": 3, "PR": 1}
    link6 = {"s": 4, "d": 2, "BW": 3, "PR": 1}

    topology_json["entity"].append(cloud_dev_1)
    topology_json["entity"].append(cloud_dev_2)
    topology_json["entity"].append(sensor_dev_1)
    topology_json["entity"].append(sensor_dev_2)
    topology_json["entity"].append(actuator_dev)
    topology_json["link"].append(link1)
    topology_json["link"].append(link2)
    topology_json["link"].append(link3)
    topology_json["link"].append(link4)
    topology_json["link"].append(link5)
    topology_json["link"].append(link6)
    # topology_json["link"].append(link4)

    return topology_json


# @profile
def main(simulated_time):

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    """
    TOPOLOGY from a json
    """
    t = Topology()
    t_json = create_json_topology()
    t.load(t_json)
    # t.write("network.gexf")
    # t.show()

    """
    APPLICATION
    """
    app = create_application()

    """
    PLACEMENT algorithm
    """
    placement = CloudPlacement(
        "onCloud")  # it defines the deployed rules: module-device
    # apply the algo on service A
    placement.scaleService({"ServiceA": 2})

    """
    POPULATION algorithm
    """
    # In ifogsim, during the creation of the application, the Sensors are assigned to the topology, in this case no. As mentioned, YAFS differentiates the adaptive sensors and their topological assignment.
    # In their case, the use a statical assignment.
    pop = Statical("Statical")
    # For each type of sink modules we set a deployment on some type of devices
    # A control sink consists on:
    #  args:
    #     model (str): identifies the device or devices where the sink is linked
    #     number (int): quantity of sinks linked in each device
    #     module (str): identifies the module from the app who receives the messages
    pop.set_sink_control(
        {"model": "actuator-device", "number": 1, "module": app.get_sink_modules()})

    # In addition, a source includes a distribution function:
    dDistribution = deterministicDistribution(name="Deterministic", time=100)
    another_distribution = deterministicDistribution(
        name="Deterministic", time=50)
    pop.set_src_control({"model": "sensor-device-1", "number": 1,
                         "message": app.get_message("M.A"), "distribution": dDistribution})
    pop.set_src_control({"model": "sensor-device-2", "number": 1,
                         "message": app.get_message("M.A"), "distribution": another_distribution})

    """--
    SELECTOR algorithm
    """
    # Their "selector" is actually the shortest way, there is not type of orchestration algorithm.
    # This implementation is already created in selector.class,called: First_ShortestPath
    selectorPath = CustomPath()

    """
    SIMULATION ENGINE
    """

    stop_time = simulated_time
    s = Sim(t, default_results_path="Results")
    s.deploy_app(app, placement, pop, selectorPath)
    s.run(stop_time, show_progress_monitor=False)
    s.draw_allocated_topology()
    # s.draw_allocated_topology() # for debugging


if __name__ == '__main__':
    #import logging.config
    #import os

    # logging.config.fileConfig(os.getcwd()+'/logging.ini')

    start_time = time.time()
    main(simulated_time=1000)

    print("\n--- %s seconds ---" % (time.time() - start_time))

    # Finally, you can analyse the results:
    # print "-"*20
    # print "Results:"
    # print "-" * 20
    m = Stats(defaultPath="Results")  # Same name of the results
    time_loops = [["M.A"]]
    m.showResults2(1000, time_loops=time_loops)

    print "\t- Network saturation -"
    print "\t\tAverage waiting messages : %i" % m.average_messages_not_transmitted()
    print "\t\tPeak of waiting messages : %i" % m.peak_messages_not_transmitted()
    print "\t\tTOTAL messages not transmitted: %i" % m.messages_not_transmitted()

    print "\n\t- Stats of each service deployed -"
    print m.get_df_modules()
    print m.get_df_service_utilization("ServiceA", 1000)
    print "\n\t- Stats of each DEVICE -"
    # print "\t- Network saturation -"
    # print "\t\tAverage waiting messages : %i" % m.average_messages_not_transmitted()
    # print "\t\tPeak of waiting messages : %i" % m.peak_messages_not_transmitted()PartitionILPPlacement
    # print "\t\tTOTAL messages not transmitted: %i" % m.messages_not_transmitted()
