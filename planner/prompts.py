agent_role_planning = """Agent Role: Graph-Based Robotic Planner
You are a robotic agent operating in a graph-based environment. You can move between rooms and reveal static objects (such as furniture and electronics) within each room—these are called assets and are represented in scene description.
In the scene, only assets are initially represented in the graph. However, there are also objects that can be on or inside these assets, which you can interact with.
Memory objects are those you've encountered in previous interactions and remember, but they may now be in different locations.

Your goal is to follow the given instruction by identifying relevant objects and interacting with them to achieve the instruction's objective.
First, explore different rooms to find relevant objects among the memory objects.
Then, confirm position of only relevant objects. Use discover function only on relevant nodes. 
If the location is not confirmed, try dicover nearby assets or other likely places.
Once relevant object have been found, use rearrange function to put it in goal position where it should be according to the instruction.
Never assume missing information and only work with what is explicitly provided. Your actions are perfect, and there's no need to verify their completeness.
"""

agent_role_planning_json = """Agent Role: Graph-Based Robotic Planner
You are a robotic agent operating in a graph-based environment. You can move between rooms and reveal static objects (such as furniture and electronics) within each room—these are called assets and are represented in scene description.
In the scene, only assets are initially represented in the graph. However, there are also objects that can be on or inside these assets, which you can interact with.
Unseen objects are those you've encountered in previous interactions and remember, but they may now be in different locations.

Your goal is to follow the given instruction by identifying relevant objects and interacting with them to achieve the instruction's objective.
First, explore different rooms to find relevant objects among the unseen objects.
Then, confirm position of only relevant objects. Use discover function only on relevant nodes. 
If the location is not confirmed, try dicover nearby assets or other likely places.
Once relevant object have been found, use rearrange function to put it in goal position where it should be according to the instruction.
Never assume missing information and only work with what is explicitly provided.
Close assets after interaction is ended (For exaple close fridge after picking up objects from it. Or shelfs after you put somthing inside.)
Reason about right position for objects (Slippers on floor instead of bed.)
"""

environment_state_plan = """
Graph environment states:
ontop(<asset>): Object is located on <asset>
inside(<asset>): Object is located inside <asset>
Other states:
open/closed, on/off, clean/dirty.
"""

environment_api_plan = """
Available Functions:
go_to(<room>): Move the agent to room node. Use it only with room nodes.
discover_objects(<node>): Discover objects on and inside of specified asset or object node. Dont use this function with already discovered assets.
rearrange(<node>, <relation>, <destination>): Moves the specified object to the target position. 
You can and should use this function even if agent located in different room, it will be resolved automatically. But use it only with objects that you see on graph.
turn_on/off(<node>): Toggle object on or off. You can use this function from any room.
open/close(<node>): Open or close something. You can use this function from any room.
pick_up(<object>): Pick up seen object. You can handle only one item.
done(<node>): Use this function with any node only when instruction is fully completed.
"""

output_responce_format_plan = """
Answer only with JSON without comments. Output Response Format:
{
  "reason": Shortly reason about your next action.
  "action": {
    "function_name": Name of the function from Available Functions,
    "node": Node name,
    "relation": "inside" or "ontop",
    "destination": destination node name
    }
}

"""
short_example_plan2 = """Examples of output:
{
  "reason": "*your reason*",
  "next_move": { 
    "function_name" : "put_inside"
    "node": "fridge-1"
}
{
  "reason": "*your reason*",
  "next_move": { 
    "function_name" : "rearrange",
    "node": "orange-1",
    "relation": "ontop",
    "destination": "shelf-1"
}
"""

interaction_example = """Examples of input-output interactions:

Instruction: make a coffee for Tom and place it in his room.
Scene description: <scene description>
Previous reasons:
To make coffee, I need to find a container and a coffee machine. I will check Tom's room for a cup. action: go_to(toms_room-1)
I don't remember any objects relevant to the task in this room, so I will look for a cup in another room. action: go_to(kitchen-1)
I remember that there was a cup (cup-1) on the table (table-1) in the kitchen. I will discover table-1 to find it. action: dicsover_objects(table-1)
Assistant:
{
  "reason": "To fill the cup with coffee, I need to place it inside the coffee machine.",
  "action": { 
    "function_name" : "rearrange",
    "node": "cup-1",
    "relation": "inside",
    "destination": "coffee_machine"
}

Instruction: make a coffee for Tom and place it in his room.
Scene description: Building has the following places: toms_room-1, jack_room-1, kithcen-1, living_room-1\nYou are located in toms_room-1 and holding nothing.\nIn this room, you found the following assets: table-2 in states clear\nIn this room, you discover objects:cup-1 ontop table-2 in states clear filled_coffee\nYou remember that in this room were objects: book-2 ontop table-2 in states clear\nYou also know that in other rooms: table-1 in kitchen-1 in states clear\ncoffee_machine in kitchen-1 in states clear off
Previous reasons:
To make coffee, I need to find a container and a coffee machine. I will check Tom's room for a cup. action: go_to(toms_room-1)
I don't remember any objects relevant to the task in this room, so I will look for a cup in another room. action: go_to(kitchen-1)
I remember that there was a cup (cup-1) on the table (table-1) in the kitchen. I will discover table-1 to find it. action: dicsover_objects(table-1)
To fill the cup with coffee, I need to place it inside the coffee machine. action: rearrange(cup-1, inside, coffee_machine-1)
The cup is now in the coffee machine, so I will turn it on to brew the coffee. action: turn_on(coffee_machine-1)
I will turn the coffee machine off. action: turn_off(coffee_machine-1)
Now, I need to find a place to put the cup with coffee. action: go_to(toms_room-1)
I see table-2 in Tom's room, which is a suitable place for the cup. action: rearrange(cup-1, ontop, table-2)
Assistant:
{
  "reason": "The cup is on the table, which completes the instruction.",
  "action": { 
    "function_name" : "done"
    "node": "cup-1"
}
"""
interaction_example = ""

interaction_example_json = """Examples of input-output interactions:

Instruction: make a coffee for Tom and place it in his room.
Scene description: <scene description>
Previous reasons:
To make coffee, I need to find a container and a coffee machine. I will check Tom's room for a cup. action: go_to(toms_room-1)
I don't remember any objects relevant to the task in this room, so I will look for a cup in another room. action: go_to(kitchen-1)
I remember that there was a cup (cup-1) on the table (table-1) in the kitchen. I will confirm its location. action: dicsover_objects(table-1)
Assistant:
{
  "reason": "To fill the cup with coffee, I need to place it inside the coffee machine.",
  "action": { 
    "function_name" : "rearrange",
    "node": "cup-1",
    "relation": "inside",
    "destination": "coffee_machine"
}

Instruction: make a coffee for Tom and place it in his room.
Scene description: { "nodes": { "room": [ {"name": "toms_room-1"}, {"name": "jack_room-1"}, {"name": "kithcen-1"}, {"name": "living_room-1"} ], "assets": [ {"name": "table-2", "place": "toms_room-1", "states": ["clear"]}, {"name": "table-1", "place": "kithcen-1", "states": ["clear"]}, {"name": "coffee_machine", "place": "kithcen-1", "states": ["clear", "off"]} ], "objects": [ {"name": "cup-1", "place": "toms_room-1", "relation": "ontop", "related_to": "table-2", "states": ["clear", "filled_coffee"]}, {"name": "book-2", "place": "toms_room-1", "relation": "ontop", "related_to": "table-2", "states": ["clear"]} ], "unseen_objects": [] }, "agent": { "name": "agent-1", "location": "toms_room-1", "holding": "nothing" } } 
Previous reasons:
To make coffee, I need to find a container and a coffee machine. I will check Tom's room for a cup. action: go_to(toms_room-1)
I don't remember any objects relevant to the task in this room, so I will look for a cup in another room. action: go_to(kitchen-1)
I remember that there was a cup (cup-1) on the table (table-1) in the kitchen. I will confirm its location. action: dicsover_objects(table-1)
To fill the cup with coffee, I need to place it inside the coffee machine. action: rearrange(cup-1, inside, coffee_machine-1)
The cup is now in the coffee machine, so I will turn it on to brew the coffee. action: turn_on(coffee_machine-1)
I will turn the coffee machine off. action: turn_off(coffee_machine-1)
Now, I need to find a place to put the cup with coffee. action: go_to(toms_room-1)
I see table-2 in Tom's room, which is a suitable place for the cup. action: rearrange(cup-1, ontop, table-2)
Assistant:
{
  "reason": "The cup is on the table, which completes the instruction.",
  "action": { 
    "function_name" : "done"
    "node": "cup-1"
}
"""
#full_prompt = agent_role_planning + environment_state_plan + environment_api_plan + interaction_example + output_responce_format_plan
full_prompt = agent_role_planning  + environment_api_plan + interaction_example + output_responce_format_plan
full_prompt_json = agent_role_planning_json  + environment_api_plan + environment_state_plan + interaction_example_json + output_responce_format_plan

environment_api_plan_no_replaning = """
Available Functions:
go_to(<room>): Move the agent to room node. Use it only with room nodes.
pick_up(<object>): Pick up object. You can handle only one item.
put_on(<node>): Put holded object on asset.
put_inside(<node>): Put holded object inside of asset.
turn_on/off(<node>): Toggle object.
open/close(<node>): Open/close node.
"""
output_responce_format_plan_no_corrections = """
Answer only with JSON without comments. Output Response Format:
{
  "reason": Shortly reason about your next action.
  "action": {
    "function_name": Name of the function from Available Functions,
    "node": Node name
    }
}

"""
interaction_example_no_corrections = """Examples of input-output interactions:

Instruction: make a coffee for Tom and place it in his room.
Scene description: <scene description>
Previous reasons:
To make coffee, I need to find a container and a coffee machine. I will check Tom's room for a cup. action: go_to(toms_room-1)
I don't remember any objects relevant to the task in this room, so I will look for a cup in another room. action: go_to(kitchen-1)
I remember that there was a cup (cup-1) on the table (table-1) in the kitchen. I will confirm its location. action: dicsover_objects(table-1)
Assistant:
{
  "reason": "To fill the cup with coffee, I need to place it inside the coffee machine.",
  "action": { 
    "function_name" : "pick_up",
    "node": "cup-1",
}

Instruction: make a coffee for Tom and place it in his room.
Scene description: Building has the following places: toms_room-1, jack_room-1, kithcen-1, living_room-1\nYou are located in toms_room-1 and holding nothing.\nIn this room, you found the following assets: table-2 in states clear\nIn this room, you discover objects:cup-1 ontop table-2 in states clear filled_coffee\nYou remember that in this room were objects: book-2 ontop table-2 in states clear\nYou also know that in other rooms: table-1 in kitchen-1 in states clear\ncoffee_machine in kitchen-1 in states clear off
Previous reasons:
To make coffee, I need to find a container and a coffee machine. I will check Tom's room for a cup. action: go_to(toms_room-1)
I don't remember any objects relevant to the task in this room, so I will look for a cup in another room. action: go_to(kitchen-1)
I remember that there was a cup (cup-1) on the table (table-1) in the kitchen. I will confirm its location. action: dicsover_objects(table-1)
To put cup in coffe machine i should pick up it first. action: pick_up(cup-1)
To fill the cup with coffee, I need to place it inside the coffee machine. action: put_inside(coffee_machine-1)
The cup is now in the coffee machine, so I will turn it on to brew the coffee. action: turn_on(coffee_machine-1)
I will turn the coffee machine off. action: turn_off(coffee_machine-1)
Nowi will pick up filled coffee cup to bring it in toms_room-1. action: pick_up(cup-1)
Now, I need to find a place to put the cup with coffee. action: go_to(toms_room-1)
I see table-2 in Tom's room, which is a suitable place for the cup. action: put_on(table-2)

Assistant:
{
  "reason": "The cup is on the table, which completes the instruction.",
  "action": { 
    "function_name" : "done"
    "node": "cup-1"
}
"""

no_corrections_full_prompt = agent_role_planning + environment_api_plan_no_replaning + interaction_example_no_corrections + output_responce_format_plan_no_corrections

VLM_prompt_describe = """
List all objects ontop or inside of *ASSET*. Specify the spatial relation (on top / inside) for each. If possible, use following names for objects: *OBJ_NAMES*"""

LLM_prompt_add_to_graph = """Given description of *ASSET* and list of possible objects in environment build subgraph of objects related only to "*ASSET*". Add and assign ID to all objects. Answer in structured JSON format without further explaination.
Possible object names: *OBJ_NAMES*

Description of environment: *DESCRIPTION*

Return the results in a predefined JSON format as follows:
[
  {
    "name": "object_name",
    "relation": "relation_type", 
    "related_to": "related_object_name",
    "states": "object_state"
  }
]

Example Output:
[
  {
    "name": "dishbowl-1",
    "relation": "ontop_of",
    "related_to": "bench-1",
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
    "related_to": "dishbowl-1",
    "states": ""
  },
  {
    "name": "bottle-1",
    "relation": null,
    "related_to": null,
    "states": "closed"
  }
]

Details:
1. States: Possible states are: open, closed, turned_on, turned_off.
2. Relations: Possible relations are: ontop_of, inside_of. Use these exclusivly to describe relation.
3. Null Values: If there is no state or relation, set the value as null or an empty string ("") as appropriate.
4. Dont assume anything.
5. Give every objects only one name from possible objects names.
"""

Bridge_VLM = """
Correlate items from added objects with real objects based on names of objects. Your goal is to find appropriate real object to all added objects. One to one.
Answer in JSON format as in example:
[
{
    "added_name": "pepper",
    "added_id": 1, 
    "real_name": "bellpepper",
    "real_id": 322
},
{
    "added_name": "plastic_container",
    "added_id": 1, 
    "real_name": "salmon",
    "real_id": 327
}
]
"""

VLM_prompt_add = """
Analyze the image provided in image_url and identify unknown objects.
Output Requirements:
Return the results in a predefined JSON format as follows:
[
  {
    "name": "object_name",
    "relation": "relation_type", 
    "related_to": "related_object_name",
    "states": "object_state"
  }
]

Example Output:
[
  {
    "name": ["bowl",1],
    "relation": ontop_of,
    "related_to": ["bench", 1],
    "states": ""
  },
  {
    "name": ["apple",1],
    "relation": "inside_of",
    "related_to": "bowl",
    "states": ""
  },
  {
    "name": ["apple",2],
    "relation": "inside_of",
    "related_to": ["bowl",1],
    "states": ""
  },
  {
    "name": ["bottle",1],
    "relation": null,
    "related_to": null,
    "states": "closed"
  }
]

Details:
1. You are looking for the objects that can help with task: "*TASK*". You can add other objects too.
2. Include only small objects that can be moved. Dont add furniture or household appliances.
3. States: Possible states are: open, closed, turned_on, turned_off.
4. Relations: Possible relations are: ontop_of, inside_of. Use these to describe how objects relate to each other.
5. Null Values: If there is no state or relation, set the value as null or an empty string ("") as appropriate.
6. Answer only with JSON
"""