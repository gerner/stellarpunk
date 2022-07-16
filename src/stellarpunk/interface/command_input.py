import curses
from typing import Tuple, Optional, Any, Sequence, Dict, Tuple, List, Mapping, Callable, Union, MutableMapping

from stellarpunk import interface, util

class CommandInput(interface.View):
    """ Command mode: typing in a command to execute. """

    CommandSig = Union[
            Callable[[Sequence[str]], None],
            Tuple[
                Callable[[Sequence[str]], None],
                Callable[[str, str], str]]
    ]

    class UserError(Exception):
        pass

    def __init__(self, *args:Any, commands:Mapping[str, CommandSig]={}, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)

        self.commands:MutableMapping[str, Callable[[Sequence[str]], None]] = {}
        self.completers:MutableMapping[str, Callable[[str, str], str]] = {}
        for c, carg in commands.items():
            if isinstance(carg, tuple):
                self.commands[c] = carg[0]
                self.completers[c] = carg[1]
            else:
                self.commands[c] = carg

        self.partial = ""
        self.command = ""

    def _command_name(self) -> str:
        i = self.command.strip(" ").find(" ")
        if i < 0:
            return self.command.strip()
        else:
            return self.command.strip()[0:i]

    def _command_args(self) -> Sequence[str]:
        return self.command.strip().split()[1:]

    def initialize(self) -> None:
        for c, f in self.interface.command_list().items():
            if c not in self.commands:
                self.commands[c] = f
        self.logger.info("entering command mode")

    def update_display(self) -> None:
        self.interface.status_message(f':{self.command}')

    def handle_input(self, key:int, dt:float) -> bool:
        if key in (ord('\n'), ord('\r')):
            self.logger.debug(f'read command {self.command}')
            self.interface.status_message()
            # process the command
            command_name = self._command_name()
            if command_name in self.commands:
                self.logger.info(f'executing {self.command}')
                try:
                    self.commands[command_name](self._command_args())
                except CommandInput.UserError as e:
                    self.logger.info(f'user error executing {self.command}: {e}')
                    self.interface.status_message(f'error in "{self.command}" {str(e)}', curses.color_pair(1))
            else:
                self.interface.status_message(f'unknown command "{self.command}" enter command mode with ":" and then "quit" to quit.', curses.color_pair(1))
            return False
        elif chr(key).isprintable():
            self.command += chr(key)
            self.partial = self.command
        elif key == curses.ascii.BS:
            self.command = self.command[:-1]
            self.partial = self.command
        elif key == curses.ascii.TAB:
            if " " not in self.command:
                self.command = util.tab_complete(self.partial, self.command, sorted(self.commands.keys())) or self.partial
            elif self._command_name() in self.completers:
                self.command = self.completers[self._command_name()](self.partial, self.command) or ""
        elif key == curses.ascii.ESC:
            self.interface.status_message()
            return False

        return True
