import os.path
import toml # type: ignore
import importlib.resources
import types
from typing import Dict, Optional, Any, List, TextIO
import contextlib

def merge(a:Dict[str, Any], b:Dict[str, Any], path:Optional[List[str]]=None) -> Dict[str, Any]:
    """ recursively merges dict b into dict a

    b[key] overrides a[key] if key present in both. raises an exception if
    b[key] and a[key] are not of the same type.

    inspired by https://stackoverflow.com/a/51653724/553580
    """

    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key].__class__ == b[key].__class__:
                a[key] = b[key]
            else:
                raise ValueError('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def dict_to_simplenamespace(d:Dict[str, Any]) -> types.SimpleNamespace:
    """ Converts a dict recursively to a SimpleNamespace. """
    d = d.copy()
    for key in d:
        if isinstance(d[key], dict):
            d[key] = dict_to_simplenamespace(d[key])

    return types.SimpleNamespace(**d)

def load_config(config_file:Optional[TextIO]=None) -> types.SimpleNamespace:
    #config = toml.loads(importlib.resources.read_text("stellarpunk.data", "config.toml"))
    config = toml.loads(importlib.resources.files("stellarpunk.data").joinpath("config.toml").read_text())
    if config_file:
        override = toml.load(config_file)
        merge(config, override)

    global Settings
    Settings = dict_to_simplenamespace(config)

    return Settings

def load_dialogs() -> Dict[str, Any]:
    #dialogs = toml.loads(importlib.resources.read_text("stellarpunk.data", "dialogs.toml"))
    dialogs = toml.loads(importlib.resources.files("stellarpunk.data").joinpath("dialogs.toml").read_text())
    return dialogs

def load_events() -> Dict[str, Any]:
    #events = toml.loads(importlib.resources.read_text("stellarpunk.data", "events.toml"))
    events = toml.loads(importlib.resources.files("stellarpunk.data").joinpath("events.toml").read_text())
    return events

def key_help(obj: object, k: str) -> str:
        try:
            return getattr(getattr(Settings.help.interface, obj.__class__.__name__).keys, k)
        except AttributeError:
            return "NO HELP"

def get_key_help(obj: object, help_key: str) -> Optional[str]:
    try:
        return getattr(getattr(Settings.help.interface, obj.__class__.__name__).keys, help_key)
    except AttributeError:
        return None

# it's ok to reload the config with a file elsewhere, but we start with the
# built-in config
Settings = load_config()
Dialogs = load_dialogs()
Events = load_events()
