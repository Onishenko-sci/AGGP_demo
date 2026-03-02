import networkx as nx
import copy
from dataclasses import dataclass
from typing import Optional, Union, Tuple

MAX_OBJECTS_DEEP = 10

@dataclass
class Action:
    function_name: str
    node:  Tuple[str, int]
    relation: Optional[str] = None
    destination: Optional[Tuple[str, int]] = None

AVAILABLE_ACTIONS = ['go_to', 'pick_up', 'put_on', 'put_inside', 'open', 'close', 'turn_on', 'turn_off', 'discover_objects']
INTERACTION_ACTIONS = ['pick_up', 'put_on', 'put_inside', 'open', 'close', 'turn_on', 'turn_off']

class SceneSimulator():
    def __init__(self, graph: nx.Graph):
        self.graph = copy.deepcopy(graph)
        self._reset_edges()
        self.actions = []

    def valid_action(self, action, node):
        if node not in self.graph:
            return False, f"Node {node} not found in the graph."
        
        if action not in AVAILABLE_ACTIONS:
            return False, f"Action {action} is wrong. No such action as {action}."
        return True, ''
    
    def valid_affordance(self, action, node_name):
        attr = self.graph.nodes[node_name]
        sayplan_name = node_name[0] +'-'+ str(node_name[1])
        if action in INTERACTION_ACTIONS:
            if attr['type'] in ['asset', 'object']  and action not in attr['affordances']:
                return False, f"Action {action}({sayplan_name}) is wrong. It is not possible to {action} {sayplan_name}."
            if attr['type'] not in ['asset','object', 'unseen_object']:
                return False, f"Action {action}({sayplan_name}) is wrong. Agent can interact only with objects and assets."
        return True, ''
        
    def correct_action(self, action: Action):
        """
        Correct chosen action or retunr feedback on it.
        """
        #Done can be called anytime
        if action.function_name == 'done':
            self.actions.append(('done', action.node))
            return self.actions, True, ''
        #Composite action
        if action.function_name == 'rearrange':
            _, success, feedback = self.correct_action(Action('pick_up', action.node))
            if not success:
                return [], False, feedback
            if action.relation == 'inside':
                _, success, feedback = self.correct_action(Action('put_inside', action.destination))
            else:
                _, success, feedback = self.correct_action(Action('put_on', action.destination))
            if not success:
                return [], False, feedback
            return self.actions, True, ''

        #Node and function filters
        valid_action, feedback = self.valid_action(action.function_name, action.node)
        if not valid_action:
            return [], False, feedback
        #Short names
        G = self.graph
        node_name = action.node
        action = action.function_name
        agent = G.nodes[('agent',1)]
        node = G.nodes[node_name]
        # Affordance filters
        satisfy_affordance, feedback = self.valid_affordance(action, node_name)
        if not satisfy_affordance:
            return [], False, feedback
        #Place, memory and access correction
        place = self.find_place(node)
        if action != 'go_to' and place != agent['location']:
            self.correct_action(Action('go_to', place))

        if action != 'go_to' and node['type'] == 'unseen_object':
            discover_objects_target, relation = self.find_seen_container(node)
            self.correct_action(Action('discover_objects', discover_objects_target))
            return self.actions, True, ''

        accesible, parent_name = self.is_accessible(node)
        if not accesible:
            self.correct_action(Action('open', parent_name))
        # Variable for correction
        target_changed = False
        sayplan_name = node_name[0] +'-'+ str(node_name[1])
        match action:
            case 'go_to':
                if agent['location'] == node_name:
                    return [], False, f"Action {action}({sayplan_name}) is wrong. Agent aldeady in room {sayplan_name}"

                if node['type'] != 'place':
                    target_changed = True
                    self.actions.append(('go_to', place))
                    agent['location'] = place
                else:
                    agent['location'] = node_name

            case 'pick_up':
                if agent['holding'] != '' and agent['holding'] != node_name:
                    return [], False, f"Action {action}({sayplan_name}) is wrong. Agent can hold only one object. Agent aldeady holding {agent['holding']}"
                if agent['holding'] == node_name:
                    target_changed = True
                    
                agent['holding'] = node_name
                node['relation'] = 'inside_hand'
                node['related_to'] = ('agent',1)

            case 'put_on':
                if agent['holding'] == '':
                    return [], False, f"Action {action}({sayplan_name}) is wrong. Agent don't holding something to put"

                G.nodes[agent['holding']]['relation'] = 'ontop_of'
                G.nodes[agent['holding']]['related_to'] = node_name
                agent['holding'] = ''

            case 'put_inside':
                if agent['holding'] == '':
                    return [], False, f"Action {action}({sayplan_name}) is wrong. Agent don't holding something to put"
                if agent['holding'] == node:
                    return [], False, f"Action {action}({sayplan_name}) is wrong. An object cannot be placed inside itself."
                if 'closed' in node['states']:
                    self.correct_action(Action('open', node_name))

                G.nodes[agent['holding']]['relation'] = 'inside_of'
                G.nodes[agent['holding']]['related_to'] = node_name
                agent['holding'] = ''

            case 'open':
                if 'open' in node['states']:
                    target_changed = True
                    pass
                else:
                    node['states'].remove('closed')
                    node['states'].append('open')
            case 'close':
                if 'closed' in node['states']:
                    target_changed = True
                    pass
                else:
                    node['states'].remove('open')
                    node['states'].append('closed')
            case 'turn_on':
                if 'on' in node['states']:
                    pass
                    target_changed = True
                elif 'off' in node['states']:
                    node['states'].remove('off')
                node['states'].append('on')

            #    if node_name[0] in ['microwave'] and 'closed' in node['states']:
            #        for obj, attr in G.nodes(data=True):
            #            if attr['type'] == 'object' and attr['related_to'] == node_name and attr['relation'] == 'inside_of':
            #                attr['states'].append('hot')

            #    if node_name[0] in ['coffee_machine']:
            #        for obj, attr in G.nodes(data=True):
            #            if attr['type'] == 'object' and attr['related_to'] == node_name and attr['relation'] == 'inside_of':
            #                attr['states'].append('filled_coffee')

            case 'turn_off':
                if 'off' in node['states']:
                    target_changed = True
                    pass
                else:    
                    node['states'].remove('on')
                    node['states'].append('off')

            case 'discover_objects':
                if node['type'] not in ['asset', 'object']:
                    return [], False, f"Action {action}({sayplan_name}) is wrong. You can discover_objects only asset and seen objects nodes."
                if 'closed' in node['states']:
                    self.correct_action(Action('open', node_name))
                    
                self.actions.append(('discover_objects', node_name))
                return self.actions, True, ''

        if not target_changed:
            self.actions.append((action, node_name))

        self._reset_edges()
        return self.actions, True, ''
    

    def is_accessible(self, node):
        """
        Object is not accessible if any 'inside_of' parent is 'closed'.
        """
        if node['type'] not in ['object', 'unseen_object']:
            return True, ''
        for _ in range(MAX_OBJECTS_DEEP):
            parent_name = node['related_to']
            parent = self.graph.nodes[parent_name]
            if parent['type']=='agent':
                return True, ''
            if 'closed' in parent['states'] and node['relation'] == 'inside_of':
                return False, parent_name
            if parent['type'] not in ['object', 'unseen_object']:
                return True, ''
            node = parent
        return True, ''
    
    def find_place(self, node):
        """Retunrs place of any type of node"""
        if node['type'] in ['place', 'scene']:
            return node
        for _ in range(MAX_OBJECTS_DEEP+1):
            if node['type'] == 'asset':
                return node['place']
            if node['type'] == 'agent':
                return node['location']
            parent_name = node['related_to']
            node = self.graph.nodes[parent_name]
    
    def find_seen_container(self, node):
        """
        Find seen parent (asset or object) to discover.
        """
        for _ in range(MAX_OBJECTS_DEEP):
            parent_name = node['related_to']
            relation = node['relation']
            node = self.graph.nodes[parent_name]
            if node['type'] in ['asset', 'object']:
                return parent_name, relation

    def _reset_edges(self):
        self.graph.remove_edges_from(list(self.graph.edges))
        for node_name in self.graph.nodes:
            self._add_edge(node_name)

    def _add_edge(self, node_name):
        node = self.graph.nodes[node_name]
        try:
            match node['type']:
                case 'place':
                    self.graph.add_edge(node_name,('scene',1))
                case 'asset':
                    self.graph.add_edge(node_name,node['place'])
                case 'object':
                    self.graph.add_edge(node_name,node['related_to'])
                case 'unseen_object':
                    self.graph.add_edge(node_name,node['related_to'])
                case 'agent':
                    self.graph.add_edge(node_name,node['location'])
                case 'scene':
                    pass
                case _:
                    print(f"Unknown node type: {node['type']}")
        except:
            raise ValueError(f"Error during edges resetting with {node_name} {node}. Check relations.")
        

if __name__ == "__main__":
    from grasif.office.office_graph import *
    test = SceneSimulator(G)
    place = ('kitchen',1)
    for node, attr in test.graph.nodes(data=True):
        if test.find_place(attr) == place:
            print(node, attr)

    print(test.correct_action(Action('rearrange',('tomato',2), "inside", ("cheese",1))))
    print(test.actions)