import networkx as nx
from copy import deepcopy
from SceneSim import SceneSimulator

base_graph = nx.Graph()
base_graph.add_node(("scene", 1), type="scene")
# Add a place
base_graph.add_node(("place", 1), type="place")
base_graph.add_node(("place", 2), type="place")
# Add an asset (e.g., fridge)
base_graph.add_node(("fridge", 1), type="asset", states=["closed"], place=("place", 1), affordances=["open", "close", "put_inside"])
base_graph.add_node(("table", 1), type="asset", states=[], place=("place", 1), affordances=["put_on"])
# Add a small object container (e.g., box) inside asset
base_graph.add_node(("box", 1), type="object", states=["closed"], relation="inside_of", related_to=("fridge", 1), affordances=["open", "close", "put_inside","put_on"])
# Add a final object inside the box as a tuple node
base_graph.add_node(("apple", 1), type="object", states=["closed"], relation="inside_of", related_to=("box", 1), affordances=["pick_up", "open"])
base_graph.add_node(("seed", 1), type="object", states=[], relation="inside_of", related_to=("apple", 1), affordances=["pick_up"])
base_graph.add_node(("banana", 1), type="object", states=[], relation="ontop_of", related_to=("box", 1), affordances=["pick_up"])
base_graph.add_node(("water", 1), type="object", states=[], relation="ontop_of", related_to=("table", 1), affordances=["pick_up"])
base_graph.add_node(("water", 2), type="object", states=[], relation="ontop_of", related_to=("table", 1), affordances=["pick_up"])
base_graph.add_node(("water", 3), type="object", states=[], relation="ontop_of", related_to=("table", 1), affordances=["pick_up"])
# Add an agent at the place
base_graph.add_node(("agent", 1), type="agent", location=("place", 2), holding="")

sim = SceneSimulator(base_graph)
actions , status, feedback = sim.correct_action('pick_up', ('seed',1))
print(status)
print(feedback)
for action in actions:
    print(action)

actions , status, feedback = sim.correct_action('pick_up', ('banana',1))
print(status)
print(feedback)
for action in actions:
    print(action)