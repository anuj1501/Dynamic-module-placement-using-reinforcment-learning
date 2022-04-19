
import pandas as pd
import os
"""
    Created on Wed Nov 22 15:03:21 2017
    @author: isaac
"""
import random

import argparse

from yafs.core import Sim

from yafs.application import Application, Message

from yafs.population import *

from yafs.selection import Selection

from yafs.topology import Topology

from simpleSelection import CustomPath

from simplePlacement import CloudPlacement

from yafs.stats import Stats

from yafs.distribution import deterministicDistribution

from yafs.utils import fractional_selectivity

import time
import numpy as np

import sys

# RANDOM_SEED = 0

sys.dont_write_bytecode = True


# @profile
def main(get_action,reward,add_time, iteration, simulated_time):

    # random.seed(RANDOM_SEED)
    # np.random.seed(RANDOM_SEED)

    """
    PLACEMENT algorithm
    """
    placement = CloudPlacement(
        "onCloud")  # it defines the deployed rules: module-device
    # apply the algo on service A
    placement.scaleService({"ServiceA": 2})

    """--
    SELECTOR algorithm
    """
    # Their "selector" is actually the shortest way, there is not type of orchestration algorithm.
    # This implementation is already created in selector.class,called: First_ShortestPath
    selectorPath = CustomPath(Selection,get_action,execution_type="dql")
    selectorPath.create_topology()
    selectorPath.set_population()

    """
    SIMULATION ENGINE
    """

    stop_time = simulated_time
    s = Sim(False,reward,selectorPath.topology, add_time,
            default_results_path="Results_" + str(iteration))
    s.deploy_app(selectorPath.app, placement, selectorPath.pop, selectorPath)
    selectorPath.init_state(s)
    s.run(stop_time, selectorPath, show_progress_monitor=False)
    # s.draw_allocated_topology()
    # s.draw_allocated_topology() # for debugging


def driver(get_action,reward):
    start_time = time.time()

    add_time = 0

    for i in range(1):

        main(get_action,reward,add_time, i, simulated_time=230)

        add_time += 100

        # print("\n--- %s seconds ---" % (time.time() - start_time))

        # Finally, you can analyse the results:
        # print "-"*20
        # print "Results:"
        # print "-" * 20
        m = Stats(defaultPath="Results_" + str(i))  # Same name of the results
        time_loops = [["M.A", "M.B"]]
        # m.showResults2(100, time_loops=time_loops)

        # print "\t- Network saturation -"
        # print "\t\tAverage waiting messages : %i" % m.average_messages_not_transmitted()
        # print "\t\tPeak of waiting messages : %i" % m.peak_messages_not_transmitted()
        # print "\t\tTOTAL messages not transmitted: %i" % m.messages_not_transmitted()

        # print "\n\t- Stats of each service deployed -"
        # print m.get_df_modules()
        # print m.get_df_service_utilization("ServiceA", 100)
        # print "\n\t- Stats of each DEVICE -"


if __name__ == '__main__':
    #import logging.config
    #import os

    # logging.config.fileConfig(os.getcwd()+'/logging.ini')

    driver()

# Results = pd.DataFrame(columns=["id", "type", "app", "module", "message", "DES.src", "DES.dst", "TOPO.src",
#                                 "TOPO.dst", "module.src", "service", "time_in", "time_out", "time_emit", "time_reception"])

# Results_link = pd.DataFrame(
#     columns=["id", "type", "src", "dst", "app", "latency", "message", "ctime", "size", "buffer"])


# for i in range(10):

#     result_temp = pd.read_csv("Results_" + str(i) + ".csv")

#     result_link_temp = pd.read_csv(
#         "Results_" + str(i) + "_link.csv")

#     Results = Results.append(result_temp, ignore_index=True)

#     Results_link = Results_link.append(result_link_temp, ignore_index=True)

#     os.remove("Results_" + str(i) + ".csv")

#     os.remove("Results_" + str(i) + "_link.csv")


# Results.to_csv("Results.csv", index=False)

# Results_link.to_csv("Results_link.csv", index=False)
