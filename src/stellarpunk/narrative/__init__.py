""" Narrative System for Stellarpunk

Handles events and interactions between characters, the player, etc. This
includes dialog, quests, some of the AI behavior.

The system is event based. We count on the game engine passing events to the
narrative Director. The Director decides who should respond to that event and
how. This might be simple notifications or messages that the player or other
characters receive. This might cause more complex behaviors by the characters.

This is a rule engine. Events come in with context (event specific, world
general) and we evaluate rules in the context of relevant characters to respond
to that event.

Rules are evaluated in the context of the event and each character. Each rule has criteria that is matched against the event/world/character context.

If the criteria match any rules, we choose one (or more?) rule to act.

Rules can write back state to the world or the character, trigger more events
and/or "do stuff". "Do stuff" might be game mechanic specific (e.g. send a
message to someone, broadcast some chatter, initiate a dialog with the player,
changing relationship, etc.)

Some motivating examples:

Tutorial
When a new game starts, we want to offer the player a tutorial on messages,
navigation, docking, mining and trade. This plays out with another character
(the tutorial guy), a station (tutorial station) and an asteroid (tutorial
asteroid). Tutorial guy sends the player a message, the player eventually has
some dialog and gets some tasks to perform. As those tasks are performed the
tutorial progresses until the player has completed the tutorial.

Mafia Reporter
When the player docks at a particular station under TBD circumstances, they get
a message from someone requesting transport to another station. They can do a
dialog, and accept (or not) the fare. When they undock they get another message
asking them to take the person somewhere else. They can do either one,
depending on what happens, they get different dialogs, different relationships
are made and more events unfold.

Comms Chatter
Occasionally an AI character will broadcast some comms chatter. Other AI
characters might respond to this creating a dialog chain. Depending on the
characters and the state different sorts of dialogs might unfold.

Violating Protected Space
The player goes near some location. They get a message that the space is
protected and they're violating it. If the player doesn't leave within some
period of time, they get a new message that they're in trouble and a bounty
hunter comes after them (perhaps immediately).

"""

from .director import EventContext, Event, CharacterCandidate, Action, Director, context
from .rule_parser import loads, loadd
