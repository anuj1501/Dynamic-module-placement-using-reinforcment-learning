"""
    This type of algorithm have two obligatory functions:

        *initial_allocation*: invoked at the start of the simulation

        *run* invoked according to the assigned temporal distribution.

"""

from yafs.placement import Placement


class CloudPlacement(Placement):
    """
    This implementation locates the services of the application in the cheapest cloud regardless of where the sources or sinks are located.

    It only runs once, in the initialization.

    """

    def initial_allocation(self, sim, app_name):
        # We find the ID-nodo/resource
        value = {"mytag": "cloud"}  # or whatever tag

        id_cluster = sim.topology.find_IDs(value)

        app = sim.apps[app_name]
        services = app.services
        # print("id_cluster")
        # print(id_cluster)
        # print("app")
        # print(app)
        # print("services")
        # print(services)
        # print(self.scaleServices)

        for module in services:
            if module in self.scaleServices:
                for rep in range(0, self.scaleServices[module]):
                    for one_id in id_cluster:
                        if True:
                            # print "node atrributes"
                            sim.topology.nodeAttributes[one_id]["services"].add(
                                module)
                            # print(sim.topology.nodeAttributes[one_id])
                            idDES = sim.deploy_module(
                                app_name, module, services[module], [one_id])

        num_services = {}
        num_sensors = {}

        for one_node in id_cluster:
            # print sim.topology.nodeAttributes[one_node]["services"]
            num_services[one_node] = len(
                sim.topology.nodeAttributes[one_node]["services"])
            num_sensors[one_node] = 0

        links = sim.topology.get_edges()
        for link in links:
            if link[1] in num_sensors.keys():
                temp_data = sim.topology.nodeAttributes[link[0]]["model"]
                if "sensor" in temp_data:
                    num_sensors[link[1]] += 1
            else:
                temp_data = sim.topology.nodeAttributes[link[1]]["model"]
                if "sensor" in temp_data:
                    num_sensors[link[0]] += 1

        for link in links:
            if link[1] in num_sensors.keys():
                temp_data = sim.topology.nodeAttributes[link[0]]["model"]
                if "sensor" in temp_data:
                    node_bandwidth = sim.topology.nodeAttributes[link[1]
                                                                 ]["device_bandwidth"]
                    temp_val = node_bandwidth / \
                        (num_services[link[1]] * num_sensors[link[1]])
                    if sim.topology.nodeAttributes[link[0]]["device_bandwidth"] > temp_val:
                        sim.topology.get_edge(
                            link)[sim.topology.LINK_BW] = temp_val
                    else:
                        sim.topology.get_edge(link)[
                            sim.topology.LINK_BW] = sim.topology.nodeAttributes[link[0]]["device_bandwidth"]
                        sim.topology.nodeAttributes[link[1]]["unitilised_bandwidth"] += temp_val - \
                            sim.topology.nodeAttributes[link[0]
                                                        ]["device_bandwidth"]

            else:
                temp_data = sim.topology.nodeAttributes[link[1]]["model"]
                if "sensor" in temp_data:
                    node_bandwidth = sim.topology.nodeAttributes[link[0]
                                                                 ]["device_bandwidth"]
                    temp_val = node_bandwidth / \
                        (num_services[link[0]] * num_sensors[link[0]])
                    if sim.topology.nodeAttributes[link[1]]["device_bandwidth"] > temp_val:
                        sim.topology.get_edge(
                            link)[sim.topology.LINK_BW] = temp_val
                    else:
                        sim.topology.get_edge(link)[
                            sim.topology.LINK_BW] = sim.topology.nodeAttributes[link[1]]["device_bandwidth"]
                        sim.topology.nodeAttributes[link[0]]["unitilised_bandwidth"] += temp_val - \
                            sim.topology.nodeAttributes[link[1]
                                                        ]["device_bandwidth"]

        # print num_services
        # print num_sensors
        # print sim.topology.get_edges()

        #print(sim.topology.get_edge((0, 3))[sim.topology.LINK_BW])

    # end function
