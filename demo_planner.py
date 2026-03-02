import matplotlib
matplotlib.use("Agg")

import networkx as nx
import matplotlib.pyplot as plt
import yaml
import os
import time
import copy
import re
import io
import random
import threading
from matplotlib.lines import Line2D
from PIL import Image

from planner.planner import Planner
from demo_env import demo_graph
from utils.generators import OpenRouterModel
from benchmarks import VirtualHome
from amb_detector.amb_detector import run_disambiguation_pipeline


# ─────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────

def load_config():
    with open("demo_config.yaml", "r") as f:
        config = yaml.safe_load(f.read())
    key = config["LLM"].get("open_router_key")
    if not key or key == "${OPEN_ROUTER_KEY}":
        key = os.getenv("OPEN_ROUTER_KEY")
    config["model"] = OpenRouterModel(
        config["LLM"]["model_name"],
        key,
        config["LLM"]["router_config"],
        config["LLM"]["short_name"],
    )
    return config


# ─────────────────────────────────────────────────────────
#  Graph helpers
# ─────────────────────────────────────────────────────────

def add_objects(gt_graph, asset, memory_graph: nx.Graph):
    """Replace unseen_object children of *asset* with actual objects from gt_graph."""
    childs = _find_all_childs(memory_graph, asset, "unseen_object")
    memory_graph.remove_nodes_from(childs)
    new_childs = _find_all_childs(gt_graph, asset, "object")
    true_nodes = [(c, gt_graph.nodes[c]) for c in new_childs]
    memory_graph.add_nodes_from(true_nodes)


def _find_all_childs(graph, asset, childs_type):
    childs = []
    visited = {asset}
    stack = [asset]
    while stack:
        node = stack.pop()
        for neighbor in graph.neighbors(node):
            if neighbor in visited:
                continue
            if graph.nodes[neighbor]["type"] not in ("unseen_object", "object"):
                continue
            visited.add(neighbor)
            stack.append(neighbor)
            if graph.nodes[neighbor]["type"] == childs_type:
                childs.append(neighbor)
    return childs


def _reset_edges(graph):
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
            raise ValueError(
                f"Error resetting edges for {node_name}. Attributes: {node}"
            )


# ─────────────────────────────────────────────────────────
#  Ambiguity checker
# ─────────────────────────────────────────────────────────

class AmbiguityChecker:
    @staticmethod
    def _format_node(node, mode):
        if mode == "simple":
            return re.sub(r"(?<!^)([A-Z])", r" \1", node[0])
        return f"{node[0]}-{node[1]}"

    def extract_objects(self, graph, mode):
        objects = []
        for node, attr in graph.nodes(data=True):
            if attr["type"] in ("unseen_object", "object"):
                objects.append(self._format_node(node, mode))
            elif attr["type"] == "asset":
                objects.append(self._format_node(node, mode))
        return objects

    def check(self, instruction, graph, mode):
        objects = self.extract_objects(graph, mode)
        results = run_disambiguation_pipeline(instruction, objects, True, True)
        info = results.get("predicates_info", {})
        conflict = info.get("conflict", {})
        variability = info.get("variability", {})
        has_ambiguity = bool(conflict or variability)
        return has_ambiguity, {"conflict": conflict, "variability": variability}


# ─────────────────────────────────────────────────────────
#  States
# ─────────────────────────────────────────────────────────

class State:
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    COMPLETED = "completed"
    RESOLVABLE_CHECK = "resolvable_check"
    AMBIGUITY_CHECK = "ambiguity_check"


# ─────────────────────────────────────────────────────────
#  Random tasks
# ─────────────────────────────────────────────────────────

SIMPLE_TASKS = [
    # Impossible
    "Dance for me.",
    "Write a Python function to calculate the Fibonacci sequence.",
    # Simple
    "Put chicken sandwich and apple juice on the countertop.",
    "Bring slippers to my bedroom.",
    "Turn on the light in the bedroom.",
    "Bring the stool to the dining room.",
    # Ambiguity
    "Put a cup on the dining table.",
    "Bring a sandwich to my work desk.",
    # Long-horizon
    "Put all plates in the dishwasher.",
    "Prepare the dining room table for a party.",
    # Long-horizon + ambiguity
    "Put the things away in the bedroom.",
]

VH_TASKS = [
    "Bring pencil to the kitchen table.",
    "Put boardgame on the table.",
    "Put jacket on the bed.",
]


# ─────────────────────────────────────────────────────────
#  Main application
# ─────────────────────────────────────────────────────────

class PlannerApp:
    """
    Central planner application.

    Exposes a generator-based ``process_message`` that yields SSE-ready
    event dicts.  A ``threading.Event`` allows the caller to interrupt
    planning from another thread (the "Stop" button).
    """

    MAX_PLANNING_STEPS = 25

    def __init__(self):
        self.config = load_config()
        self.model = self.config["model"]

        self.dataset = VirtualHome
        self.dataset.get_tasks()
        self.task_data = self.dataset.tasks[11]

        self.gt_graph = copy.deepcopy(demo_graph)
        self._filter_graph()
        self.mode = "simple"

        self.graph_history: list[Image.Image] = []
        self.graph_index: int = 0

        self.planner = self._create_planner()
        self.ambiguity_checker = AmbiguityChecker()

        self.state = State.IDLE
        self.instruction = ""
        self.clarifying_question = ""
        self.conversation_log: list[tuple[str, str]] = []

        self._stop_event = threading.Event()
        self._generation = 0

        # Initial graph snapshot
        self._add_graph(self._draw_graph(full=True))

    # ── Factory ────────────────────────────────────────────

    def _create_planner(self):
        return Planner(
            starting_graph=copy.deepcopy(self.gt_graph),
            LLM=self.model,
            VLM=self.model,
            ablation_mode="default",
        )

    # ── Public API ─────────────────────────────────────────

    def stop(self):
        """Signal the planner to stop at the next safe checkpoint."""
        self._stop_event.set()

    def reset_env(self, env: str):
        """Reset the environment and planner."""
        if env == "simple":
            self.gt_graph = copy.deepcopy(demo_graph)
            self.mode = "simple"
        else:
            self.gt_graph = copy.deepcopy(self.task_data["init"])
            self._filter_graph()
            self.mode = "vh"

        self.graph_history = []
        self.graph_index = 0
        self.planner = self._create_planner()
        self._add_graph(self._draw_graph(full=True))
        self.state = State.IDLE
        self.instruction = ""
        self.clarifying_question = ""
        self.conversation_log = []

    def get_random_task(self) -> str:
        tasks = SIMPLE_TASKS if self.mode == "simple" else VH_TASKS
        return random.choice(tasks)

    # ── Streaming message processor ────────────────────────

    def process_message(self, message: str):
        """
        Generator that yields event dicts for each step of the conversation.
        Designed to run in a background thread; the HTTP layer wraps it
        into an SSE stream.
        """
        self._generation += 1
        my_gen = self._generation
        self._stop_event.clear()

        # Log user message
        self.conversation_log.append(("user", message))

        yield {"type": "status", "state": State.RUNNING}
        time.sleep(0.5)

        if self.state == State.AWAITING_CLARIFICATION:
            yield from self._handle_clarification(message, my_gen)
        elif self.state == State.STOPPED:
            yield from self._handle_correction(message, my_gen)
        else:
            yield from self._handle_new_instruction(message, my_gen)

    # ── Handlers ───────────────────────────────────────────

    def _is_stopped(self, my_gen: int) -> bool:
        return self._stop_event.is_set() or self._generation != my_gen

    def _handle_new_instruction(self, message: str, my_gen: int):
        # 1. Capability check
        yield {"type": "status", "state": State.RESOLVABLE_CHECK}
        time.sleep(1)

        if self._is_stopped(my_gen):
            self.state = State.STOPPED
            yield {"type": "status", "state": State.STOPPED}
            return

        if not self._is_capable(message):
            yield {
                "type": "message",
                "text": (
                    "Sorry, this is beyond my current capabilities. "
                    "I can help with pick-and-place tasks. "
                    "Please rephrase or try a different task."
                ),
            }
            self.state = State.IDLE
            yield {"type": "status", "state": State.IDLE}
            return

        if self._is_stopped(my_gen):
            self.state = State.STOPPED
            yield {"type": "status", "state": State.STOPPED}
            return

        # 2. Ambiguity check
        yield {"type": "status", "state": State.AMBIGUITY_CHECK}
        time.sleep(1)

        self.instruction = message
        ambiguous, details = self.ambiguity_checker.check(
            message, self.planner.memory_graph, self.mode
        )

        if ambiguous:
            yield {"type": "message", "text": f"Ambiguity detected: {details}"}
            self.conversation_log.append(("agent", f"Ambiguity detected: {details}"))
            time.sleep(1)
            question = self._create_clarify_question(message, details)
            self.clarifying_question = question
            yield {"type": "message", "text": question}
            self.conversation_log.append(("agent", question))
            self.state = State.AWAITING_CLARIFICATION
            yield {"type": "status", "state": State.AWAITING_CLARIFICATION}
            return

        yield {"type": "status", "state": State.RUNNING}
        time.sleep(1)

        # 3. Plan
        self.planner.set_task(self.instruction)
        yield from self._run_planning_loop(my_gen)

    def _handle_clarification(self, message: str, my_gen: int):
        refined = self._resolve_ambiguity(message, self.instruction)
        self.instruction = refined
        yield {
            "type": "message",
            "text": f'Instruction refined: "{refined}". Starting planning.',
        }
        self.conversation_log.append(("agent", f'Instruction refined: "{refined}"'))
        time.sleep(1)

        self.planner.set_task(self.instruction)
        yield from self._run_planning_loop(my_gen)

    def _handle_correction(self, message: str, my_gen: int):
        yield {"type": "status", "state": State.RUNNING}
        time.sleep(1)

        # Use LLM to refine the original instruction with the correction
        refined = self._refine_instruction(self.instruction, message)
        yield {
            "type": "message",
            "text": f'Instruction updated: "{refined}"',
        }
        self.instruction = refined
        time.sleep(1)

        # Start fresh planning with the refined instruction
        self.planner.set_task(self.instruction)
        yield from self._run_planning_loop(my_gen)

    def _run_planning_loop(self, my_gen: int):
        self.state = State.RUNNING

        for i in range(self.MAX_PLANNING_STEPS):
            if self._is_stopped(my_gen):
                self.state = State.STOPPED
                yield {"type": "status", "state": State.STOPPED}
                return

            print(f"Planning step {i + 1}")
            actions = self.planner.get_action()

            for act, node in actions:
                if act == "discover_objects":
                    add_objects(self.gt_graph, node, self.planner.memory_graph)
                    self.planner._reset_edges()

            self._add_graph(self._draw_graph())

            reason = self.planner.reasons[-1] if self.planner.reasons else ""
            action = self.planner.actions[-1] if self.planner.actions else ""

            yield {"type": "message", "text": f"{reason}\nAction: {action}"}
            yield {
                "type": "graph_updated",
                "index": self.graph_index + 1,
                "total": len(self.graph_history),
            }

            time.sleep(1)

            if self.planner.goal_reached:
                yield {"type": "message", "text": "Goal marked as completed."}
                self.state = State.COMPLETED
                yield {"type": "status", "state": State.COMPLETED}
                return

        yield {"type": "message", "text": "Action limit reached. Planning stopped."}
        self.state = State.COMPLETED
        yield {"type": "status", "state": State.COMPLETED}

    # ── LLM helpers ────────────────────────────────────────

    def _is_capable(self, instruction: str) -> bool:
        prompt = (
            "You are a capability analyzer for a manipulation robot. "
            "A robotic agent has memory, great perception and can make decisions "
            "and resolve ambiguity by asking a human when needed.\n"
            "Your task is to determine if the given instruction can be executed "
            "using only these low-level actions:\n"
            "- go_to\n- pick_up\n- open\n- close\n"
            "- put_on\n- put_inside\n- turn_on\n- turn_off\n\n"
            "Examples:\n"
            '- "Put apple in microwave" -> True (uses pick_up, open, put_inside, close)\n'
            '- "Bring something to eat on table" -> True (uses pick_up, put_on)\n'
            '- "Turn on light" -> True (uses turn_on)\n'
            '- "Bring sandwich to my work desk" -> True (uses pick_up, go_to, put_on)\n'
            '- "Prepare dining room table for a party." -> True (uses pick_up, go_to, put_on)\n'
            '- "Write Fibonacci python function" -> False (requires coding)\n'
            '- "Make a dance for me" -> False (requires complex coordinated movements)\n'
            '- "Spread strawberry yogurt on bread" -> False (requires spreading/smearing)\n\n'
            f'Instruction: "{instruction}"\n'
            'First reason about your capability to follow this instruction, '
            'then respond with "True" or "False".'
        )
        answer, _ = self.model.message_answer(prompt)
        answer_lower = answer.strip().lower()
        print(f"Capability check for: {instruction}")
        print(f"LLM answer: {answer}")
        return "true" in answer_lower

    def _create_clarify_question(self, instruction: str, details) -> str:
        print("LLM call: create clarifying question")
        prompt = (
            f'The user provided the instruction: "{instruction}"\n'
            f"This instruction is ambiguous because: {details}\n"
            "Generate a clarifying question for the user to resolve this "
            "ambiguity. Respond only with the question."
        )
        answer, _ = self.model.message_answer(prompt)
        return answer

    def _resolve_ambiguity(self, user_response: str, initial_instruction: str) -> str:
        print("LLM call: resolve ambiguity")
        conversation = [
            {"role": "user", "content": initial_instruction},
            {"role": "system", "content": self.clarifying_question},
            {"role": "user", "content": user_response},
            {
                "role": "system",
                "content": (
                    "Rewrite the instruction to be unambiguous based on the "
                    "user's response. Respond only with the new instruction."
                ),
            },
        ]
        answer, _ = self.model.conversation_answer(conversation)
        return answer

    def _refine_instruction(self, original: str, correction: str) -> str:
        """Use LLM to merge the original instruction with a user correction,
        using the full conversation context so references like 'both' resolve."""
        print("LLM call: refine instruction with correction")

        # Build conversation context from log
        context_lines = []
        for role, text in self.conversation_log[-10:]:  # last 10 exchanges
            prefix = "User" if role == "user" else "Agent"
            context_lines.append(f"{prefix}: {text}")
        context_str = "\n".join(context_lines)

        prompt = (
            "Here is the conversation between a user and a robotic assistant:\n"
            f"{context_str}\n\n"
            f'The robot was executing the instruction: "{original}"\n'
            f'The user stopped the robot and said: "{correction}"\n\n'
            "Using the conversation context above, rewrite the instruction "
            "incorporating the user's correction into a single clear, "
            "unambiguous instruction. Resolve any references (e.g. 'both', "
            "'the other one') using the conversation history.\n"
            "Respond only with the new instruction, nothing else."
        )
        answer, _ = self.model.message_answer(prompt)
        return answer.strip().strip('"')

    # ── Graph management ───────────────────────────────────

    def _add_graph(self, image: Image.Image):
        self.graph_history.append(image)
        self.graph_index = len(self.graph_history) - 1

    def get_current_graph(self) -> Image.Image | None:
        if not self.graph_history:
            return None
        return self.graph_history[self.graph_index]

    def step_left(self):
        if self.graph_history:
            self.graph_index = max(0, self.graph_index - 1)
        return self.get_current_graph()

    def step_right(self):
        if self.graph_history:
            self.graph_index = min(len(self.graph_history) - 1, self.graph_index + 1)
        return self.get_current_graph()

    def get_counter(self) -> str:
        total = max(len(self.graph_history), 1)
        return f"{self.graph_index + 1}/{total}"

    def graph_to_bytes(self, img: Image.Image | None = None) -> bytes:
        if img is None:
            img = self.get_current_graph()
        if img is None:
            return b""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ── Graph drawing ──────────────────────────────────────

    def _draw_graph(self, full: bool = False) -> Image.Image:
        graph = self.planner.memory_graph
        if full:
            G = graph
        else:
            relevant_nodes = self.planner._relevant_nodes()
            allowed = set(relevant_nodes)
            allowed.add(("scene", 1))
            nodes_to_draw = [n for n in graph.nodes() if n in allowed]
            if not nodes_to_draw:
                raise ValueError("No relevant nodes found in graph")
            G = graph.subgraph(nodes_to_draw).copy()

        _reset_edges(G)

        if self.mode == "simple":
            fig = self._simple_graph(G)
        else:
            fig = self._default_graph(G)

        return self._fig_to_pil(fig)

    @staticmethod
    def _fig_to_pil(fig) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    # ── Simple-mode graph ──────────────────────────────────

    def _simple_graph(self, G):
        scale = 1.3
        font = 12

        groups = {}
        for n, attr in G.nodes(data=True):
            t = attr.get("type", "unknown")
            groups.setdefault(t, []).append(n)

        style = {
            "scene":         {"color": "#B700FF", "shape": "o", "size": 3000},
            "agent":         {"color": "#1f77b4", "shape": "P", "size": 2000},
            "place":         {"color": "#2ca02c", "shape": "p", "size": 2200},
            "asset":         {"color": "#ff7f0e", "shape": "o", "size": 2200},
            "object":        {"color": "#d62728", "shape": "s", "size": 1700},
            "unseen_object": {"color": "#7f7f7f", "shape": "s", "size": 1700},
        }

        fig = plt.figure(
            figsize=(int(10 * scale), int(7 * scale)), facecolor="none"
        )
        ax = plt.gca()
        ax.set_facecolor("none")

        legend_handles = []
        for t, st in style.items():
            handle = Line2D(
                [0], [0],
                marker=st["shape"],
                color="white",
                markerfacecolor=st["color"],
                markersize=10,
                linestyle="None",
                label=t,
            )
            legend_handles.append(handle)
        if legend_handles:
            legend = ax.legend(handles=legend_handles, loc="lower right", frameon=False)
            for text in legend.get_texts():
                text.set_color("white")

        labels = {}
        for n, attr in G.nodes(data=True):
            s = n[0]
            s = re.sub(r" ", r"\n", s)
            s = re.sub(r"(?<!^)([A-Z])", r"\n\1", s)
            label = s
            if attr["type"] == "asset":
                for state in attr["states"]:
                    label += f"\n({state})"
            labels[n] = label

        pos = nx.kamada_kawai_layout(G)

        for t, nodes in groups.items():
            st = style.get(t, style["object"])
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=st["color"],
                node_size=st["size"],
                node_shape=st["shape"],
                linewidths=0.5,
                edgecolors="black",
            )

        nx.draw_networkx_labels(
            G, pos, labels, font_size=font, font_weight="bold", font_color="white"
        )
        nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.7, edge_color="gray")

        edge_labels = {
            (u, v): d.get("relation", "")
            for u, v, d in G.edges(data=True)
            if d.get("relation")
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.axis("off")
        plt.tight_layout()
        return fig

    # ── Default-mode graph ─────────────────────────────────

    def _default_graph(self, G):
        pos = nx.kamada_kawai_layout(G)
        groups = {}
        for n, attr in G.nodes(data=True):
            t = attr.get("type", "unknown")
            groups.setdefault(t, []).append(n)

        style = {
            "scene":         {"color": "#B700FF", "shape": "o", "size": 800},
            "agent":         {"color": "#1f77b4", "shape": "P", "size": 800},
            "place":         {"color": "#2ca02c", "shape": "s", "size": 800},
            "asset":         {"color": "#ff7f0e", "shape": "o", "size": 500},
            "object":        {"color": "#d62728", "shape": "d", "size": 400},
            "unseen_object": {"color": "#7f7f7f", "shape": "d", "size": 300},
            "unknown":       {"color": "#8c564b", "shape": "o", "size": 300},
        }

        fig = plt.figure(figsize=(10, 7), facecolor="none")
        ax = plt.gca()
        ax.set_facecolor("none")

        for t, nodes in groups.items():
            st = style.get(t, style["unknown"])
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=st["color"],
                node_size=st["size"],
                node_shape=st["shape"],
                linewidths=0.5,
                edgecolors="black",
            )

        legend_handles = []
        for t, st in style.items():
            if t == "unknown":
                continue
            handle = Line2D(
                [0], [0],
                marker=st["shape"],
                color="white",
                markerfacecolor=st["color"],
                markersize=10,
                linestyle="None",
                label=t,
            )
            legend_handles.append(handle)
        if legend_handles:
            legend = ax.legend(handles=legend_handles, loc="lower right", frameon=False)
            for text in legend.get_texts():
                text.set_color("white")

        labels = {}
        for n, attr in G.nodes(data=True):
            base = f"{n[0]}-{n[1]}"
            t = attr.get("type")
            labels[n] = base
            if t == "asset":
                has_obj = False
                for nbr in G.neighbors(n):
                    if G.nodes[nbr].get("type") in ("object", "unseen_object"):
                        has_obj = True
                        break
                if not has_obj:
                    continue

        nx.draw_networkx_labels(
            G, pos, labels, font_size=8, font_weight="bold", font_color="white"
        )
        nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.7, edge_color="gray")

        edge_labels = {
            (u, v): d.get("relation", "")
            for u, v, d in G.edges(data=True)
            if d.get("relation")
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.axis("off")
        plt.tight_layout()
        return fig

    # ── Graph filter ───────────────────────────────────────

    def _filter_graph(self):
        remove = [
            "photoframe", "orchid", "shower", "light_switch", "powersocket",
            "knifeblock", "ceilinglamp", "walllamp", "curtain", "window",
            "cpuscreen", "hanger", "drawing", "stovefan", "ceilingfan",
            "clothespile", "controller", "foodfood", "phone", "vase",
        ]
        be_deleted = [
            node
            for node, _ in self.gt_graph.nodes(data=True)
            if node[0] in remove
        ]
        self.gt_graph.remove_nodes_from(be_deleted)
        _reset_edges(self.gt_graph)


# ─────────────────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = PlannerApp()
    app.reset_env("simple")
    img = app.graph_history[0]
    img.save("output.png")
    print("Graph saved to output.png")
