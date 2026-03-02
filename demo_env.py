import networkx as nx

ROOMS = [
    ('kitchen', 1), 
    ('diningroom', 1),
    ('bedroom', 1),
    ('hallway', 1),
    ('bathroom',1)
]

ASSETS = [
    #Diningroom
    (('Table', 1), ('diningroom', 1), [], ['put_on'], []),
    (('Chair', 1), ('diningroom', 1), [], ['put_on'], []),
    (('Bench', 1), ('diningroom', 1), [], ['put_on'], []),
    (('TvStand', 1), ('diningroom', 1), ['closed'], ['open', 'close', 'put_on','put_inside'], []),
    (('Tv', 1), ('diningroom', 1), ['off'], ['turn_on', 'turn_off'], []),
    (('DiningRoomFloor', 1), ('diningroom', 1), [ ], [ 'put_on'], []),
    #Kitchen
    (('Fridge', 1), ('kitchen', 1), ['closed', ], ['open', 'close', 'put_inside'], []),
    (('Dishwasher', 1), ('kitchen', 1), ['closed', 'off', ], ['open', 'close', 'turn_on', 'turn_off', 'put_on', 'put_inside'], []),
#    (('Microwave', 1), ('kitchen', 1), ['off', 'closed'], ['open', 'close', 'turn_on', 'turn_off', 'put_on', 'put_inside'], []),
    (('Countertop', 1), ('kitchen', 1), ['closed', ], ['open', 'close', 'put_on', 'put_inside'], []),
    (('KitchenFloor', 1), ('kitchen', 1), [ ], [ 'put_on'], []),
    (('Plate stand', 1), ('kitchen', 1), ['open'], ['open', 'close', 'put_inside'], []),
    #Bedroom
    (('Bed', 1), ('bedroom', 1), [], ['put_on'], []),
    (('WorkTable', 1), ('bedroom', 1), [], ['put_on'], []),
    (('Computer', 1), ('bedroom', 1), ['off'], [], []),
    (('Lamp', 1), ('bedroom', 1), ['off'] , ['turn_on', 'turn_off'], []),
    (('Shelf', 1), ('bedroom', 1), ['closed'] , ['open', 'close','put_inside'], []),
    (('BedroomFloor', 1), ('bedroom', 1), [ ], [ 'put_on'], []),
    #Hallway
    (('ShoeRack', 1), ('hallway', 1), [], ['put_on', 'put_inside'], []),
    (('Hanger', 1), ('hallway', 1), [], ['put_on'], []),
    (('HallwayFloor', 1), ('hallway', 1), [], ['put_on'], []),
    #Bathroom
    (('Toilet', 1), ('bathroom', 1), ['closed'], ['open', 'close', 'put_inside'], []),
    (('Sink', 1), ('bathroom', 1), ['off'], ['turn_on', 'turn_off', 'put_inside'], []),
    (('Shower', 1), ('bathroom', 1), ['off'], ['turn_on', 'turn_off', 'put_inside'], []),
    (('BathroomFloor', 1), ('bathroom', 1), [], ['put_on'], [])
]

# Objects (name, relation, related, [states], [affordances], [properties] )
OBJECTS = [
    #Dining room
    (('Cake', 1), 'ontop_of', ('Table', 1), [], ['pick_up'], []),
    (('WhitePlate', 1), 'ontop_of', ('Table', 1), [], ['pick_up','put_on'], []),
    (('RedPlate', 1), 'ontop_of', ('Table', 1), [], ['pick_up','put_on'], []),
    (('BigPlate', 1), 'ontop_of', ('Table', 1), [], ['pick_up','put_on'], []),
    (('Towel', 1), 'ontop_of', ('TvStand', 1), [], ['pick_up'], []),
    #Hallway
    (('Slippers', 1), 'inside_of', ('ShoeRack', 1), [], ['pick_up'], []),
    (('Boots', 1), 'inside_of', ('ShoeRack', 1), [], ['pick_up'], []),
    (('Сoat', 1), 'ontop_of', ('Hanger', 1), [], ['pick_up'], []),
#    (('Windbreaker', 1), 'ontop_of', ('Hanger', 1), [], ['pick_up'], []),
    (('Stool', 1), 'ontop_of', ('HallwayFloor', 1), [], ['pick_up', 'put_on'], []),
    #Kitchen
    (('CupWithCofee', 1), 'ontop_of', ('Countertop', 1), [], ['pick_up'], ['blue']),
    (('CupWithTea', 1), 'ontop_of', ('Countertop', 1), [], ['pick_up'], ['blue']),
#    (('bowl', 1), 'inside_of', ('Dishwasher', 1), [], ['pick_up'], ['blue']),
    (('FishSandwich', 1), 'inside_of', ('Fridge', 1), [], ['pick_up'], []),
    (('ChikenSandwich', 1), 'inside_of', ('Fridge', 1), [], ['pick_up'], []),
    (('OrangeJuice', 1), 'inside_of', ('Fridge', 1), [], ['pick_up'], []),
    (('AppleJuice', 1), 'inside_of', ('Fridge', 1), [], ['pick_up'], []),
    #Bedroom
#    (('Tshirt', 1), 'inside_of', ('Shelf', 1), [], ['pick_up'], []),
    (('Shirt', 1), 'inside_of', ('Shelf', 1), [], ['pick_up'], []),
    (('Shorts', 1), 'ontop_of', ('Bed', 1), [], ['pick_up'], []),
    (('Socks', 1), 'ontop_of', ('BedroomFloor', 1), [], ['pick_up'], []),
    #Bathroom
    (('Soap', 1), 'ontop_of', ('Sink', 1), [], ['pick_up'], []),
 #   (('Toothbrush', 1), 'inside_of', ('Sink', 1),[], ['pick_up'], []),
    (('Shampoo', 1), 'inside_of', ('Shower', 1), [], ['pick_up'], []),
]

def add_edges(graph):
    try:
        for node_name in graph.nodes:
            node = graph.nodes[node_name]
            match node["type"]:
                case "place":
                    graph.add_edge(node_name, ("scene", 1))
                case "asset":
                    graph.add_edge(node_name, node["place"])
                case "object":
                    graph.add_edge(node_name, node["related_to"])
                case "unseen_object":
                    graph.add_edge(node_name, node["related_to"])
                case "agent":
                    graph.add_edge(node_name, node["location"])
                case "scene":
                    pass
                case _:
                    print(f"Unknown node type: {node['type']}")
    except Exception as e:
        raise ValueError(f"Error during reseting edges. When edge from {node_name} added. Attributes are: {node}")
    

def create_office_graph():
    """Initializes and builds the office scene graph."""
    graph = nx.Graph()
    graph.add_node(('scene', 1), name='SayPlan Office', type='scene')

    # Add places (offices and rooms) and connect them to the scene
    for place in (ROOMS):
        graph.add_node(place, type='place')

    # Add assets and connect them to their places
    for asset, place, states, affordances, properties in ASSETS:
        graph.add_node(asset, place=place, states=states, affordances=affordances, properties=properties, type='asset')

    # Add objects and connect them to their related assets/objects
    for obj, relation, related_to, states, affordances, properties in OBJECTS:
        graph.add_node(obj, relation=relation, related_to=related_to, states=states, affordances=affordances, properties=properties, type='object')

    # Add the agent and connect it to its location
    agent_node = ('agent', 1)
    agent_location = ('hallway', 1)
    graph.add_node(agent_node, location=agent_location, holding='', type='agent')
    add_edges(graph)
    return graph

# Initialize the graph by calling the function
demo_graph = create_office_graph()