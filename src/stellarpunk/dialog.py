""" Dialog system for Stellarpunk """

from typing import Sequence, Dict, Any

from stellarpunk import config

class DialogChoice:
    def __init__(self, text:str, node_id:str):
        self.text = text
        self.node_id = node_id

class DialogNode:
    def __init__(self, node_id:str, text:str, choices:Sequence[DialogChoice], terminal:bool=False):
        self.node_id = node_id
        self.text = text
        self.choices = choices
        self.terminal = terminal

class DialogGraph:
    def __init__(self, root_id:str, nodes:Sequence[DialogNode]):
        self.root_id = root_id
        self.nodes = {x.node_id:x for x in nodes}

def load_dialog_node(dialog_data:Dict[str, Any]) -> DialogNode:
    if dialog_data.get("terminal", False):
        return DialogNode(
            dialog_data["node_id"],
            dialog_data["text"],
            [],
            True
        )
    else:
        return DialogNode(
                dialog_data["node_id"],
                dialog_data["text"],
                [DialogChoice(x["text"], x["node_id"]) for x in dialog_data["choices"]],
        )

def load_dialog(dialog_id:str) -> DialogGraph:
    dialog_data = config.Dialogs[dialog_id]

    return DialogGraph(dialog_data["root_id"], [load_dialog_node(x) for x in dialog_data["nodes"]])
