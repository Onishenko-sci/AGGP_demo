import networkx as nx
import json
from collections import defaultdict

from .prompts import VLM_prompt_describe, LLM_prompt_add_to_graph

MAX_OBJECTS_ID = 100

class GraphAugmentation:
    def __init__(
    self,
    starting_graph: nx.Graph,
    VLM,
    ):
        self.memory_graph = starting_graph
        self.vlm = VLM

        nodes_to_change = [
            node
            for node, attr in self.memory_graph.nodes(data=True)
            if attr["type"] == "unseen_object" and attr.get("relation") != "inside_hand"
        ]
        # Save list of possible names for objects
        self.possible_objects = set()
        for node in nodes_to_change:
            self.possible_objects.add(node[0])
        # Save dictionary of objects states for augmentation module
        self.prior_attr = {}
        for node, attr in self.memory_graph.nodes(data=True):
            self.prior_attr[node[0]] = attr

        self.names_counter = {}
        self._count_names()
        self.log = {'augmentation_tokens': 0}
        
    def describe_process_image(self, asset, image, add_prior = False):
        #Image description
        vlm_prompt = self._gather_describe_prompt(asset, add_prior=add_prior)
        description = self._generator(vlm_prompt, image)
        #Format to json
        format_prompt = self._gather_format_prompt(asset, description)
        answer = self._generator(format_prompt)
        #Parse filter and normilize objects
        objects = self._parse_objects(answer)
        filtered = self._normilize_objects(objects, asset)
        #Delete unseen nodes
        self._delete_objects(asset)
        #Add new nodes
        self._add_objects_from_dict(filtered)
        #Update IDs in graph and reset edges
        self._update_ids()
        self._reset_edges()
        #Return added objects
        added_objects = self._find_all_childs(asset, 'object')
        return self.memory_graph, added_objects

    def _add_objects_from_dict(self, objects: list[dict]):
        for obj in objects: 
            name = obj['node'][0]
            #New nodes get ID+100 to deviate from old objects when merged.
            new_id = obj['node'][1] + MAX_OBJECTS_ID
            self.memory_graph.add_node(
                (name, new_id),
                relation=obj["relation"],
                related_to= obj['related_to'],
                states=obj['states'],
                affordances=obj['affordances'],
                properties=obj["properties"],
                type="object"
            )
        return

    def _generator(self, message, image=None):
        answer, tokens = self.vlm.message_answer(message=message,image=image)
        self.log['augmentation_tokens'] += tokens
        return answer

    def _gather_describe_prompt(self, asset, add_prior=False):
        vlm_prompt = VLM_prompt_describe.replace("*ASSET*", asset[0]).replace("*OBJ_NAMES*", ", ".join(self.possible_objects))
        if add_prior:
            childs = self._find_all_childs(asset, 'unseen_object')
            prev_objects = {}
            for child in childs:
                if child[0] not in prev_objects:
                    prev_objects[child[0]] = 1
                else:
                    prev_objects[child[0]] += 1
            vlm_prompt += "\nPreviously, the following objects were spotted: "
            for obj in prev_objects.keys():
                vlm_prompt += f"{prev_objects[obj]} of {obj}, "
            vlm_prompt = vlm_prompt[:-2]
            vlm_prompt += ". These objects can now be removed or new ones added — please proceed carefully."
        return vlm_prompt

    def _gather_format_prompt(self, asset, description):
        prompt = LLM_prompt_add_to_graph
        prompt = prompt.replace("*ASSET*", f"{asset[0]}-{asset[1]}")
        prompt = prompt.replace("*OBJ_NAMES*", ", ".join(self.possible_objects))
        prompt = prompt.replace("*DESCRIPTION*", description)
        return prompt

    def _parse_objects(self, answer):
        start_index = answer.find("[")
        end_index = answer.rfind("]")
        json_part = answer[start_index : end_index + 1]
        try:
            objects = json.loads(json_part)
        except json.JSONDecodeError as e:
            print("Invalid JSON data:\n", e)
            return []
        
        def parse_node(node : str):
            if node is None:
                return None
            node = node.replace(' ', '_')
            name = node.split('-')
            if len(name) > 1 and name[1].isdigit():
                return (name[0],int(name[1]))
            return (name[0], None)
        
        parsed_objects = []
        for obj in objects:
            parsed_objects.append(
                {
                    "node": parse_node(obj.get("name", "Unknown")),
                    "relation": obj.get("relation", None),
                    "related_to": parse_node(obj.get("related_to", None)),
                    "states": obj.get("states", ""),
                    "properties": obj.get("properties", ""),
                    "affordances": obj.get("affordances", "")
                }
            )
        return parsed_objects
    
    def _normilize_objects(self, objects, asset):
        norm_objects = []
        new_names_counter = {}
        possible_positions = set()
        possible_positions.add(None)
        possible_positions.add(asset)
        assets = self._assets_from_same_room()
        for obj in objects:
            name = obj['node'][0]
            #Filters
            if name in assets:
                continue

            #Count added objects
            if name in new_names_counter:
                new_names_counter[name] += 1
            else:
                new_names_counter[name] = 1
            #Normilizations
            if obj['relation'] not in ['ontop_of', 'inside_of', None]:
                obj['relation'] = None

            if obj['related_to'] not in possible_positions:
                obj['related_to'] = None

            if obj['node'][1] == None:
                
                obj['node'] = (obj['node'][0], new_names_counter[name])

            if obj['relation'] == None:
                obj['relation'] = 'ontop_of'

            if obj['related_to'] == None:
                obj['related_to'] = asset
            if obj['related_to'][1] == None:
                obj['related_to'][1] = new_names_counter[obj['related_to'][0]]

            if obj['states'] == "":
                obj['states'] = self.prior_attr[name]['states'] if name in self.prior_attr else []
            elif isinstance(obj['states'], str):
                obj['states'] = [obj['states']]

            if obj['affordances'] == "":
                obj['affordances'] = self.prior_attr[name]['affordances'] if name in self.prior_attr else ["pick_up", 'put_on', 'put_inside']

            possible_positions.add(obj['node'])
            norm_objects.append(obj)
            
        return norm_objects

    def _assets_from_same_room(self):
        assets = set()
        for node_name in self.memory_graph:
            node = self.memory_graph.nodes[node_name]
            if node['type'] in ['asset'] and node['place'] == self.memory_graph.nodes[('agent',1)]['location']:
                assets.add(node_name[0])
        return assets

    def _delete_objects(self, node_name):
        neighbors = list(self.memory_graph.neighbors(node_name))
        for neighbor in neighbors:
            neighbor_type = self.memory_graph.nodes[neighbor].get("type")
            if (
                neighbor_type in ["unseen_object", "object"]
                and self.memory_graph.nodes[neighbor].get("related_to") == node_name
            ):
                self._delete_objects(neighbor)
                self.memory_graph.remove_node(neighbor)
                if neighbor[0] in self.names_counter:
                    self.names_counter[neighbor[0]] -= 1
    
    def _update_ids(self):
        """
        Recalculates ID for all objects in graph in encreasing order with prioretization for objects. States affordances and predicates states same.

        Example:
        apple-1, apple-3, apple-103 unseen object apple-5

        Become

        apple-1, apple-2, apple-3. unseen object apple-4.
        """
        # Divide by prefix
        prefix_groups = defaultdict(list)
        for node in self.memory_graph.nodes:
            prefix, num = node
            prefix_groups[prefix].append((node, num))
        # Create map from old ID to new
        mapping = {}
        for prefix, nodes in prefix_groups.items():
            nodes_sorted = sorted(nodes, key=lambda x: x[1])
            nodes_sorted = sorted(nodes_sorted, key=lambda x: self.memory_graph.nodes[x[0]]['type'] == 'unseen_object')
            for new_num, (old_node, _) in enumerate(nodes_sorted, start=1):
                new_node = (prefix, new_num)
                mapping[old_node] = new_node
        # Relabel edges
        for node, attr in self.memory_graph.nodes(data=True):
            if attr['type'] in ['object', 'unseen_object'] and attr['related_to'] in mapping:
                attr['related_to'] = mapping[attr['related_to']]
            if attr['type'] == 'agent' and attr['holding'] in mapping:
                attr['holding'] = mapping[attr['holding']]
        # Relabel nodes
        self.memory_graph = nx.relabel_nodes(self.memory_graph, mapping)
        return

    def _reset_edges(self):
        graph = self.memory_graph
        graph.remove_edges_from(list(graph.edges))
        for node_name in graph.nodes:
            node = graph.nodes[node_name]
            try:
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
        return
    
    def _find_all_childs(self, asset, c_type):
        """
        DFS search for childs with type.
        """
        childs = []
        visited = set([asset])
        stack = [asset]
        while stack:
            node = stack.pop()
            for neighbor in self.memory_graph.neighbors(node):
                if neighbor in visited:
                    continue
                if self.memory_graph.nodes[neighbor]['type'] not in ['unseen_object', 'object']:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
                if self.memory_graph.nodes[neighbor]['type'] == c_type:
                    childs.append(neighbor)
        return childs
    
    def _count_names(self):
        for node in self.memory_graph.nodes():
            key = node[0]
            if key in self.names_counter:
                self.names_counter[key] += 1
            else:
                self.names_counter[key] = 1

    def _process_nodes(self, nodes):
        for node in nodes:
            self.memory_graph.add_node(
                node[0],
                relation=node[1]["relation"],
                related_to=node[1]["related_to"],
                states=node[1]["states"],
                affordances=node[1]["affordances"],
                properties=node[1]["properties"],
                type=node[1]["type"],
            )
        self._reset_edges()


if __name__ == "__main__":
    def show_nodes(graph, asset):
        childs = test._find_all_childs(asset, 'object')
        for node, attr in graph.nodes(data=True):
            if node in childs:
                print(node, attr)

    VLM = ''
    #from benchmarks.grasif.office.office_graph import *
    #test = GraphAugmentation(G, VLM)
    #asset = ('desk',2)
    #show_nodes(test.memory_graph, asset)
    test_str = '' + \
"""
Hello! Here is objects from image:
[
    {
        "name": "dishbowl-2",
        "relation": "ontop_of",
        "related_to": "where?-1",
        "states": ""
    },
    {
        "name": "apple-1",
        "relation": "inside_of",
        "related_to": "dishbowl-1",
        "states": ""
    },
    {
        "name": "apple-2",
        "relation": "inside_of",
        "related_to": "dishbowl-4",
        "states": ["closed"]
    },
    {
        "name": "bottle-1",
        "relation": null,
        "related_to": null,
        "states": "closed"
    },
    {
        "name": "qwe",
        "relation": null,
        "related_to": null,
        "states": ["123","321"]
    },
    {
        "name": "qwe",
        "relation": null,
        "related_to": null,
        "states": ""
    },
    {
        "name": "qwe",
        "relation": null,
        "related_to": "qwe-1",
        "states": ""
    },
    {
        "name": "123+-6",
        "relation": "123",
        "related_to": "132+-6",
        "states": ""
    }
]
"""
    objects = test._parse_objects(test_str)
    test._delete_objects(asset)
    filtered = test._normilize_objects(objects, asset)
    test._add_objects_from_dict(filtered)
    test._update_ids()
    test._reset_edges()
    added_objects = test._find_all_childs(asset, 'object')
    print("Added objects:")
    print(added_objects)
    show_nodes(test.memory_graph, asset)