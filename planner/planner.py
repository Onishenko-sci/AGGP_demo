import networkx as nx
from typing import Literal
import json
import re

from .prompts import *
from .SceneSim import SceneSimulator, Action
from .Augmentator import GraphAugmentation

MAX_OBJECTS_DEEP = 10
DEFAULT_MAX_FEEDBACK_STEPS = 3
REASONS_TO_SHOW = 10

class Planner:
    """
    Planner integrates a memory graph with language and vision-language models to plan
    actions based on a given task instruction.

    Attributes:
        memory_graph (nx.Graph): A copy of the initial memory graph.
        conversation_history (list): Stores the history of conversations with the LLM.
        goal_reached (bool): Flag indicating if the goal has been reached.
        max_feedback_steps (int): Maximum number of feedback iterations allowed.
        feedback (str): Latest feedback from the LLM.
        first_prompt (bool): Flag to determine if the full prompt should be used initially.
        interactions_memory (list): Log of nodes that have been interacted with.
        n_relevant_nodes (int): Counter for the number of relevant nodes.
        names_counter (dict): Dictionary to count occurrences of names.
        log (dict): Dictionary logging instruction, actions, and VLM logs.
        client: Client for interacting with LLM/VLM services.
        llm_model_name (str): The name of the language model.
        vlm_model_name (str): The name of the vision-language model.
        vlm (callable): Function to interact with the VLM; defaults to _default_vlm if not provided.
        llm (callable): Function to interact with the LLM; defaults to _default_llm if not provided.
    """

    def __init__(
        self,
        starting_graph: nx.Graph,
        LLM,
        VLM,
        ablation_mode: Literal['default', 'json', 'no_corrections', 'json_links', 'full_graph', 'no_memory'] = 'default',
        max_feedback_steps=DEFAULT_MAX_FEEDBACK_STEPS
    ):
        """
        Initialize the Planner.

        Args:
            starting_graph (nx.Graph): The initial memory graph.
            client (Any): Client for interacting with LLM/VLM services.
            llm (str): Language model name.
            vlm (str): Vision-language model name.
            llm_f (function): Custom LLM function. Defaults to None.
            vlm_f (function): Custom VLM function. Defaults to None.
        """
        self.memory_graph = starting_graph.copy()
        self.conversation_history = []
        self.goal_reached = True
        self.max_feedback_steps =  max_feedback_steps
        self.ablation_mode = ablation_mode
        self.feedback = ""
        self.replaning = False
        self.interactions_memory = []
        self.n_relevant_nodes = 0
        self.reasons = []
        self.actions = []
        self.looked = []

        self.log = {"instruction": "", "actions": [], "vlm_log": [], "tokens": 0, 'augmentation_tokens': 0}
        self.previous_instruction = ""
        self.previous_actions = []
        #LLM handling
        self.llm = LLM

        # Mark all objects as unseen_objects
        nodes_to_change = [
            node
            for node, attr in self.memory_graph.nodes(data=True)
            if attr["type"] == "object" and attr.get("relation") != "inside_hand"
        ]
        new_attrs = {node: {"type": "unseen_object"} for node in nodes_to_change}
        nx.set_node_attributes(self.memory_graph, new_attrs)
        self._reset_edges()

        self.priors = True
        if self.ablation_mode == 'no_priours':
            self.priors = False

        #Augmentaton init
        self.augmentator = GraphAugmentation(self.memory_graph, VLM)

    def set_task(self, task: str) -> None:
        """
        Set a new task for the planner. Reset conversation history with LLM.
        Saves previous instruction context for continuity.

        Args:
            task (str): The instruction or goal.
        """
        # Save previous instruction context
        if hasattr(self, 'instruction') and self.instruction:
            self.previous_instruction = self.instruction
            self.previous_actions = list(self.actions) if self.actions else []
        else:
            self.previous_instruction = ""
            self.previous_actions = []

        self.instruction = task
        self.log["instruction"] = task
        self.goal_reached = False
        self.conversation_history = []
        self.reasons = []
        self.actions = []
        self.replaning = False

    def get_action(self):
        """
        Generate the next action based on the current state and feedback.

        Returns:
        The approved action and its target node, or (None, None) if goal is reached or max attempts exceeded.
        """
        for i in range(self.max_feedback_steps):
            answer = self._prompt_llm()
            reason, action, parser_feedback = self._parse_action(answer)
            #print(answer)
            if not action:
                self.feedback = "Wrong JSON format. " + parser_feedback
                continue
            #print("LLM chosen action:", action.function_name, action.node)
            
            approved, corrected_actions, feedback = self._correct_action(action)
            if self.ablation_mode == 'no_corrections':
                return [(action.function_name, action.node)]
            #print('----')
            #print(feedback)
            #print('----')
            if approved:
                self._update_interactions_memory(reason, action, corrected_actions)
                self.memory_graph = self._update_graph(corrected_actions)
                self._log_actions(i,action,answer)
                #Flush replaning conversation history
                self.conversation_history = []
                self.replaning = False
                return corrected_actions
            #Iterative replaning on next step
            self.feedback = feedback
            self.replaning = True

        print("Max feedback iterations reached!")
        if action is not None:
            self._log_actions(i, action, answer)
        self.goal_reached = True
        return []
    
    def _log_actions(self, i, action, answer):
        action_log = {
            "action": action.function_name,
            "node": action.node,
            "replan_tries": i,
            "answer": answer,
            "feedback": self.feedback,
        }
        self.log["actions"].append(action_log)
        return
    
    def _update_interactions_memory(self, reason, action, corrected_actions):
        """
        Format action and reason and save to use in prompt gathering.
        """
        reason_str = reason
        action_node = action.node[0]+ '-' +str(action.node[1])
        action_str = f"{action.function_name}({action_node})"
        #Not standart action for rearrange
        if action.function_name == 'rearrange':
            action_str = f"rearrange({action.node[0]}-{action.node[1]}, {action.relation}, {action.destination[0]}-{action.destination[1]})"
        #If action corrected to discover, reason and action need to be changed. 
        discover_action = False
        for act, act_node in corrected_actions:
            if act == 'discover_objects':
                discover_action = True
                discovered = act_node
        if discover_action and action.function_name != 'discover_objects':
            discovered_name = discovered[0]+ '-' +str(discovered[1])
            action_str = f"discover_objects({discovered_name})"
            reason_str = f'I remember relevant node on {discovered_name}. So i will discover {discovered_name} to find it.'
        #Save into memory to add in prompt later
        self.reasons.append(reason_str)
        self.actions.append(action_str)
        if discover_action:
            self.looked.append(discovered)
        #Update intercations memory
        for act, node in corrected_actions:
            if act not in ["go_to", "pick_up", "done"]:
                self.interactions_memory.append(node)
            if act == "done":
                self.goal_reached = True
        return

    def _prompt_llm(self):
        prompt = ""
        graph = ""

        # Build relevant memory graph.
        relevant_nodes = self._relevant_nodes()
        match self.ablation_mode:
            case 'default':
                graph +=  self._graph_to_human_str(relevant_nodes)
            case 'json':
                graph +=  self._graph_to_str(relevant_nodes)
            case 'json_links':
                graph +=  self._graph_to_str(relevant_nodes, edges=True)
            case 'full_graph':
                relevant_nodes = self.memory_graph.nodes
                graph +=  self._graph_to_human_str(relevant_nodes)
            case _:
                graph +=  self._graph_to_human_str(relevant_nodes)
        #Format previous actions.
        prev_actions = []
        for i in range(len(self.reasons)):
            prev_actions.append(f"{self.reasons[i]} action: {self.actions[i]}")
        #Gather prompt
        if not self.replaning:
            if self.ablation_mode in ['json', 'json_links']:
                prompt += full_prompt_json
            elif self.ablation_mode == 'no_corrections':
                prompt += no_corrections_full_prompt
            else:
                prompt += full_prompt
            prompt += "Instruction:" + self.instruction
            if self.previous_instruction:
                prompt += f"\n\nPrevious task context: Previously you worked on: '{self.previous_instruction}'."
                if self.previous_actions:
                    prompt += f" Actions completed: {', '.join(self.previous_actions[-5:])}"
            prompt += "\nScene description:\n"  + graph
            prompt += "\n\nPrevious reasons:\n" + "\n".join(prev_actions[-REASONS_TO_SHOW:])

        prompt += "\n\nFeedback:" + self.feedback
        self.feedback = ""
        answer= self._generator(prompt)
        return answer
    
    def _relevant_nodes(self):
        """
        Return list of previously interacted objects and assets and objects and assets that in save room with agent.
        """
        relevant_nodes = []
        agent_location = self.memory_graph.nodes[("agent", 1)]["location"]
        for node, attr in self.memory_graph.nodes(data=True):
            #Previously interacted objects and assets
            if node in self.interactions_memory:
                relevant_nodes.append(node)
                continue
            #Assets  in the same room with agent
            if attr["type"] == "asset" and attr["place"] == agent_location:
                relevant_nodes.append(node)
            #Objects in the same room with agent
            elif attr["type"] == "unseen_object":
                obj_place, chain = self._find_location(node)
                if obj_place == agent_location:
                    relevant_nodes.extend([n for n in chain if n not in relevant_nodes])
            #Other agents seen objects and rooms in environment
            elif attr["type"] in ["place", "object", "agent"]:
                relevant_nodes.append(node)
        return relevant_nodes

    def _find_location(self, node):
        '''
        Find unseen object place.
        '''
        chain = [node]
        parent = self.memory_graph.nodes[node]["related_to"]
        for _ in range(MAX_OBJECTS_DEEP):
            parent_attr = self.memory_graph.nodes[parent]
            if parent_attr["type"] == "asset":
                return parent_attr["place"], chain
            chain.append(parent)
            parent = parent_attr["related_to"]
        raise ValueError(f"Parent asset for node {node} not found within {MAX_OBJECTS_DEEP} steps.")

    def _known_assets(self):
        known_assets = []
        data = "\nKnown objects:\n"
        # Collect nodes by type
        for node, attr in self.memory_graph.nodes(data=True):
            if (
                attr["type"] == "asset"
                and attr["place"] == self.memory_graph.nodes[("agent", 1)]["location"]
            ):
                data += f"{{\"name\": [{node[0]},{node[1]}], \"states\": {attr['states']}}}\n"
                known_assets.append(node)

        for node, attr in self.memory_graph.nodes(data=True):
            if attr["type"] == "object" and attr["related_to"] in known_assets:
                data += f"{{\"name\": [{node[0]},{node[1]}], \"relation\": {attr['relation']}, \"related_to\": {attr['related_to']}}}\n"
        return data

    def _parse_action(self, answer):
        start_index = answer.find("{")
        end_index = answer.rfind("}")
        json_part = answer[start_index : end_index + 1]
        feedback = ""
        try:
            # Parse the JSON string into a dictionary
            instruction_data = json.loads(json_part)
        except json.JSONDecodeError:
            feedback = "Failed to decode JSON"
            return None, None, feedback

        # Проверка на наличие 'command' и его тип
        command = instruction_data.get("action")
        if not isinstance(command, dict):
            feedback = "action field is missing or is not a JSON object."
            return None, None, feedback

        reason = instruction_data.get("reason")
        if not isinstance(reason, str):
            print("Model dont add reason!")
            reason = ''


        command_name = command.get("function_name")
        if not isinstance(command_name, str):
            feedback = "function_name is missing or is not a string object."
            return None, None, feedback
        

        node_name = command.get("node")
        if not isinstance(node_name, str):
            feedback = "node is missing or is not a string object."
            return None, None, feedback
        
        match = re.match(r"^([a-zA-Z0-9_]+)-(\d+)$", node_name)
        if match:
            node_name = (match.group(1), int(match.group(2)))
        else:
            feedback = "Wrong node name format."

        action = Action(function_name=command_name, node=node_name)

        if command_name == 'rearrange':
            relation = command.get("relation")
            if not isinstance(relation, str):
                feedback = "relation for rearrange is missing or is not a string object."
                return None, None, feedback
            action.relation = relation

            target_node = command.get("destination")
            if not isinstance(target_node, str):
                feedback = "destination field is missing or is not a string object."
                return None, None, feedback
       
            match = re.match(r"^([a-zA-Z0-9_]+)-(\d+)$", target_node)
            if match:
                action.destination = (match.group(1), int(match.group(2)))
            else:
                action.destination = (target_node, 1)
                feedback = "Wrong target node name format."

        return reason, action, feedback

    def _correct_action(self, action):
        sim = SceneSimulator(self.memory_graph.copy())
        actions, success, feedback = sim.correct_action(action)
        if not success:
            return False, [], feedback
        
        node = action.node
        for act in actions:
            if act[0] == 'discover_objects':
                if node in self.looked:
                    return False, [], f"Action {action}({node[0]}-{node[1]}) is wrong. You already discover objects from {node[0]}-{node[1]}."
        
        return True, actions, ""

    def _update_graph(self, corrected_actions):
        """
        Returns graph with simulated actions.
        """
        sim = SceneSimulator(self.memory_graph)
        for act, node in corrected_actions:
            sim.correct_action(Action(act,node))
        return sim.graph

    def _generator(self, message, image=None):
        self.conversation_history.append(
            {"role": "user", "content": message}
        )
        answer, tokens = self.llm.conversation_answer(
            conversation=self.conversation_history,
        )
        self.log['tokens'] += tokens

        self.conversation_history.append(
            {"role": "assistant","content": answer}
        )
        return answer

    def _graph_to_str(self, relevant_nodes, edges=False):
        def out(node):
            return f"{node[0]}-{node[1]}"
        graph = self.memory_graph
        json_data = {
            "nodes": {
                "room": [],
                "asset": [],
                "objects": [],
                "unseen_objects": [],
                "agent": [],
            }
        }
        # Collect nodes by type
        self.n_relevant_nodes = len(relevant_nodes)
        for node, attr in graph.nodes(data=True):
            if node not in relevant_nodes:
                continue
            short_node = out(node)
            node_type = attr.get("type")
            if node_type == "place":
                json_data["nodes"]["room"].append({"id": short_node})
            elif node_type == "agent":
                json_data["nodes"]["agent"].append(
                    {
                        "id": short_node,
                        "location": out(attr['location']),
                        "holding": attr.get("holding", ""),
                    }
                )
            elif node_type == "asset":
                json_data["nodes"]["asset"].append(
                    {
                        "id": short_node,
                        "inspected": True if node in self.looked else False,
                        "located": out(attr["place"]),
                        "states": attr.get("states", ""),
                    }
                )
            elif node_type == "object":
                json_data["nodes"]["objects"].append(
                    {
                        "id": short_node,
                        "inspected": True if node in self.looked else False,
                        "relation": attr.get("relation", ""),
                        "related_to": out(attr["related_to"]),
                        "states": attr.get("states", ""),
                    }
                )
            elif node_type == "unseen_object":
                json_data["nodes"]["unseen_objects"].append(
                    {
                        "id": short_node,
                        "relation": attr.get("relation", ""),
                        "related_to": out(attr["related_to"]),
                        "states": attr.get("states", ""),
                    }
                )
        if edges:
            json_data["edges"]=[]
            for u, v in graph.edges():
                json_data["links"].append(f"{u[0]+'-'+ str(u[1])}<->{v[0] + '-'+  str(v[1])}")

        # Convert dictionary to JSON string
        json_str = json.dumps(json_data)
        #json_str = json.dumps(json_data).replace('{', '\n{').replace('\"objects\":','\n\"objects\":' ).replace('\"asset\":','\n\"asset\":' ).replace('\"agent\":','\n\"agent\":' ).replace('\"unseen_objects\":','\n\"unseen_objects\":' )
        return json_str

    def _graph_to_human_str(self, relevant_nodes):
        def format_node(node):
            return f"{node[0]}-{node[1]}"
        
        graph = self.memory_graph
        agent = graph.nodes[("agent", 1)]
        self.n_relevant_nodes = len(relevant_nodes)
        # Building overview
        places = [format_node(node) for node, attr in graph.nodes(data=True) if attr["type"] == "place"]
        description = f"Building has the following places: {', '.join(places)}"
        # Agent status
        holding = agent["holding"] or "nothing"
        location = format_node(agent["location"])
        description += f"\n\nYou are located in {location} and holding {holding}."

        # Assets in the current room
        assets = []
        for node in relevant_nodes:
            asset = graph.nodes[node]
            if asset.get('type') == 'asset' and asset['place'] == agent['location']:
                discovered = ' (already discovered)' if node in self.looked else ''
                state_description = f" in states {' '.join(asset['states'])}" if asset['states'] else ""
                asset_description = f"{format_node(node)}{discovered}{state_description}"
                assets.append(asset_description)
        if assets:
            description += f"\n\nIn this room, you found the following assets:\n" + "\n".join(assets)
        
        # Objects in the current room
        objects = []
        printed_objects = []
        for node in relevant_nodes:
            current = graph.nodes[node]
            if current["type"] == "object":
                for _ in range(MAX_OBJECTS_DEEP):
                    if current["type"] == "object":
                        current = graph.nodes[current["related_to"]]
                    else:
                        break
                if current['type'] == 'agent' or current["place"] == agent["location"]:
                    obj = graph.nodes[node]
                    discovered = ' (already discovered)' if node in self.looked else ''
                    state_description = f" in states {' '.join(obj['states'])}" if obj['states'] else ""
                    objects.append(
                        f"{format_node(node)}{discovered} {obj['relation'][:-3] if current['type'] != 'agent' else obj['relation']} {format_node(obj['related_to'])}{state_description}"
                    )
                    printed_objects.append(node)
        if objects:
            description += f"\n\nYou also found and can interact with following objects:\n" + "\n".join(objects)

        # Unseen objects in the current room
        unseen_objects = []
        for node in relevant_nodes:
            node_data = graph.nodes[node]
            if node_data['type'] == 'unseen_object':
                relation = node_data['relation'][:-3]
                related_to = format_node(node_data['related_to'])
                state_description = f" in states {' '.join(node_data['states'])}" if node_data['states'] else ""
                unseen_objects.append(f"{format_node(node)} {relation} {related_to}{state_description}")

        if unseen_objects and self.ablation_mode!= 'no_memory':
            description += f"\n\nYou remember that in this room were objects:\n" + "\n".join(unseen_objects)
        
        # Other rooms
        other_assets_objects = []
        for node in relevant_nodes:
            node_data = graph.nodes[node]
            if node_data["type"] == "asset" and node_data["place"] != agent["location"]:
                state_description = f" in states {' '.join(node_data['states'])}" if node_data['states'] else ""
                discovered = ' (already discovered)' if node in self.looked else ''
                other_assets_objects.append(
                    f"{format_node(node)}{discovered} in {format_node(node_data['place'])}{state_description}"
                )
            elif node_data["type"] == "object" and node not in printed_objects:
                state_description = f" in states {' '.join(node_data['states'])}" if node_data['states'] else ""
                discovered = ' (already discovered)' if node in self.looked else ''
                other_assets_objects.append(
                    f"{format_node(node)}{discovered} {node_data['relation'][:-3]} {format_node(node_data['related_to'])}{state_description}"
                )
        if other_assets_objects:
            description += f"\n\nYou also know that in other rooms:\n" + "\n".join(other_assets_objects)
        
        return description

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

    def augment_graph(self, image):
        print(f"Provide image for {self.looked[-1]}")
        self.memory_graph, added_objects = self.augmentator.describe_process_image(self.looked[-1], image, add_prior=self.priors)

        if added_objects:
            joined = ", ".join(f"{o[0]}-{o[1]}" for o in added_objects)
            self.feedback = f"You discover {joined}."
        else:
            self.feedback = "You discover nothing."
        return added_objects