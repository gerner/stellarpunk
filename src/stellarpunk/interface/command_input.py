import curses
from typing import Tuple, Optional, Any, Sequence, Dict, Tuple, List, Mapping, Callable, Union, MutableMapping

from stellarpunk import interface, util

class CommandHistory:
    """ Readlines-like command history editor.

    Model is that you've got a history of commands, you are editing an entry in it.
    The last entry is replaced by the command you enter. """

    def __init__(self) -> None:
        self._history:List[str] = [""]
        self._history_index = 0

    def _get_command(self) -> str:
        return self._history[self._history_index]

    def _set_command(self, value:str) -> None:
        self._history[self._history_index] = value

    command = property(fget=_get_command, fset=_set_command)

    def initialize_command(self) -> None:
        self._history_index = len(self._history)-1
        self._history[self._history_index] = ""

    def enter_command(self) -> None:
        """ makes sure the latest command is saved to history

        most recent command duplicates are removed """

        if len(self._history) > 1:
            if self._history[-2] != self.command:
                self._history[-1] = self.command
                self._history.append("")
            else:
                self._history[-1] = ""
        else:
            self._history.append("")

    def prev_command(self) -> str:
        self._history_index -= 1
        if self._history_index < 0:
            self._history_index = 0
        return self._history[self._history_index]

    def next_command(self) -> str:
        self._history_index += 1
        if self._history_index >= len(self._history):
            self._history_index = len(self._history)-1
        return self._history[self._history_index]

shared_history = CommandHistory()

class UserError(Exception):
    pass

class CommandInput(interface.View):
    """ Command mode: typing in a command to execute. """

    def __init__(self, *args:Any, commands:Mapping[str, interface.CommandBinding]={}, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.commands = commands

        self.partial = ""
        self._command_history = shared_history

        self.fast_render = True

    def _get_command(self) -> str: return self._command_history.command
    def _set_command(self, value:str) -> None: self._command_history.command = value
    command = property(fget=_get_command, fset=_set_command)

    def _command_name(self) -> str:
        """ Returns just the first word of the input command. """
        i = self.command.strip(" ").find(" ")
        if i < 0:
            return self.command.strip()
        else:
            return self.command.strip()[0:i]

    def _command_args(self) -> Sequence[str]:
        return self.command.strip().split()[1:]

    def initialize(self) -> None:
        self.logger.info("entering command mode")

    def focus(self) -> None:
        super().focus()
        self._command_history.initialize_command()

    def update_display(self) -> None:
        self.interface.status_message(f':{self.command}', cursor=True)

    def handle_input(self, key:int, dt:float) -> bool:
        if key in (ord('\n'), ord('\r')):
            self.logger.debug(f'read command {self.command}')
            self.interface.status_message()
            # process the command
            command_name = self._command_name()
            if command_name in self.commands:
                self._command_history.enter_command()
                self.logger.info(f'executing {self.command}')
                try:
                    self.commands[command_name](self._command_args())
                except UserError as e:
                    self.logger.info(f'user error executing {self.command}: {e}')
                    self.interface.status_message(f'error in "{self.command}" {str(e)}', curses.color_pair(1))
            else:
                self.interface.status_message(f'unknown command "{self.command}" enter command mode with ":" and then "quit" to quit.', curses.color_pair(1))
            self.interface.close_view(self)
        elif key == curses.KEY_UP:
            self._command_history.prev_command()
        elif key == curses.KEY_DOWN:
            self._command_history.next_command()
        elif chr(key).isprintable():
            self.command += chr(key)
            self.partial = self.command
        elif key == curses.ascii.BS:
            self.command = self.command[:-1]
            self.partial = self.command
        elif key == curses.ascii.TAB:
            if " " not in self.command:
                self.command = util.tab_complete(self.partial, self.command, sorted(self.commands.keys())) or self.partial
            else:
                self.command = self.commands[self._command_name()].complete(self.partial, self.command)
        elif key == curses.ascii.ESC:
            self.interface.status_message()
            self.interface.close_view(self)
        else:
            return False

        return True
