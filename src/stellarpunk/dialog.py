""" Dialog system for Stellarpunk """

from typing import Sequence, Dict, Any, Optional

from stellarpunk import config


class DialogChoice:
    def __init__(self, text: str, node_id: str, speaker_flags: Sequence[str], flags: Sequence[str]) -> None:
        self.text = text
        self.node_id = node_id
        self.speaker_flags = speaker_flags
        self.flags = flags


class DialogNode:
    def __init__(self, node_id:str, text:str, choices:Sequence[DialogChoice], terminal:bool, speaker_flags: Sequence[str], flags: Sequence[str]) -> None:
        self.node_id = node_id
        self.text = text
        self.choices = choices
        self.terminal = terminal
        self.speaker_flags = speaker_flags
        self.flags = flags


class DialogGraph:
    def __init__(self, dialog_id:str, root_id:str, nodes:Sequence[DialogNode]):
        self.dialog_id = dialog_id
        self.root_id = root_id
        self.nodes = {x.node_id:x for x in nodes}


def load_dialog_node(dialog_data:Dict[str, Any]) -> DialogNode:
    return DialogNode(
        dialog_data["node_id"],
        dialog_data["text"],
        [DialogChoice(x["text"], x["node_id"], x.get("speaker_flags", []), x.get("flags", [])) for x in dialog_data.get("choices", [])],
        dialog_data.get("terminal", False),
        dialog_data.get("speaker_flags", []),
        dialog_data.get("flags", []),
    )


def load_dialog(dialog_id:str) -> DialogGraph:
    dialog_data = config.Dialogs[dialog_id]

    return DialogGraph(
        dialog_id,
        dialog_data["root_id"],
        [load_dialog_node(x) for x in dialog_data["nodes"]],
    )
