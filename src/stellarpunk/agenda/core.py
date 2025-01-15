""" Agenda items for characters, reflecting activites they are involved in. """

import uuid
import abc
from typing import Any, Type, Optional

from stellarpunk import core
from stellarpunk.core import combat

class Agendum(core.AbstractAgendum, abc.ABC):

    def __init__(self, gamestate:core.Gamestate, *args:Any, **kwargs:Any) -> None:
        super().__init__(*args, **kwargs)
        self.gamestate = gamestate
        self.started_at = -1.0
        self.stopped_at = -1.0

    def sanity_check(self) -> None:
        super().sanity_check()
        assert(self.character.entity_id in self.gamestate.entities)

    def register(self) -> None:
        self.gamestate.register_agendum(self)

    def unregister(self) -> None:
        self.gamestate.unregister_agendum(self)

    def start(self) -> None:
        assert(self.started_at < 0.0)
        self.stopped_at = -1.0
        self.started_at = self.gamestate.timestamp
        self._start()

    def pause(self) -> None:
        super().pause()
        self.gamestate.unschedule_agendum(self)

    def stop(self) -> None:
        assert(self.started_at >= 0.0)
        assert(self.stopped_at < 0.0 or self.stopped_at == self.gamestate.timestamp)
        self._stop()
        self.stopped_at = self.gamestate.timestamp
        self.gamestate.unschedule_agendum(self)


class EntityOperatorAgendum(core.SectorEntityObserver, Agendum):
    """ Represents an agenda item that only works for a character who operates
    a sector entity, i.e. a captain. """

    @classmethod
    def create_eoa[T:EntityOperatorAgendum](cls:Type[T], craft: core.CrewedSectorEntity, *args: Any, **kwargs: Any) -> T:
        a = cls.create_agendum(*args, **kwargs)
        a.craft=craft
        return a

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.craft:core.CrewedSectorEntity = None # type: ignore

    def _start(self) -> None:
        if self.character.location is None:
            raise ValueError(f'{self.character.short_id()} tried to operate {self.craft.short_id()} but they are nowhere')
        if self.character.location != self.craft:
            raise ValueError(f'{self.character.short_id()} tried to operate {self.craft.short_id()} but they are on {self.character.location.short_id()}')
        if self.craft.captain != self.character:
            raise ValueError(f'{self.character.short_id()} tried to operate {self.craft.short_id()} but they are not the captain')
        self.craft.observe(self)

    def _stop(self) -> None:
        self.craft.unobserve(self)

    # core.SectorEntityObserver
    @property
    def observer_id(self) -> uuid.UUID:
        return self.agenda_id

    def entity_destroyed(self, entity:core.SectorEntity) -> None:
        if entity == self.craft:
            self.stop()

class CaptainAgendum(core.OrderObserver, EntityOperatorAgendum):
    """ Core behavior necessary for running/maintaining a ship. Mostly this
    does nothing, but watches for certain scenarios where a captain's
    intervention is necessary. """

    def __init__(self, *args: Any, enable_threat_response:bool=True, start_transponder:bool=False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enable_threat_response = enable_threat_response
        self.threat_response:Optional[combat.FleeOrder] = None
        self._start_transponder = start_transponder

    def _start(self) -> None:
        # become captain before underlying start so we'll be captain by that point
        self.craft.captain = self.character
        super()._start()
        for a in self.character.agenda:
            if a != self and isinstance(a, CaptainAgendum):
                raise ValueError(f'{self.character.short_id()} already had a captain agendum: {a}')

        self.craft.sensor_settings.set_transponder(self._start_transponder)

    def _stop(self) -> None:
        super()._stop()
        self.craft.captain = None
        #TODO: kill other EOAs?

    def sanity_check(self) -> None:
        super().sanity_check()
        if self.threat_response:
            assert self.threat_response.order_id in self.gamestate.orders
            assert not self.threat_response.is_complete()

    # core.OrderObserver
    def order_complete(self, order:core.Order) -> None:
        if order == self.threat_response:
            self.threat_response = None
            for a in self.character.agenda:
                if isinstance(a, EntityOperatorAgendum):
                    a.unpause()

    def order_cancel(self, order:core.Order) -> None:
        if order == self.threat_response:
            self.threat_response = None
            for a in self.character.agenda:
                if isinstance(a, EntityOperatorAgendum):
                    a.unpause()

    # core.SectorEntityObserver
    def entity_targeted(self, craft:core.SectorEntity, threat:core.SectorEntity) -> None:
        assert craft == self.craft
        assert self.craft.sector
        if not self.enable_threat_response:
            return

        # ignore if we're already handling threats
        if self.threat_response:
            return

        # determine in threat is hostile
        hostile = combat.is_hostile(self.craft, threat)
        # decide how to proceed:
        if not hostile:
            return
        # if first threat, pause other ship-operating activities (agenda), start fleeing
        self.logger.debug(f'{self.craft.short_id} initiating defensive maneuvers against threat {threat}')
        #TODO: is it weird for one agendum to manipulate another?
        for a in self.character.agenda:
            if isinstance(a, EntityOperatorAgendum):
                a.pause()

        # engage in defense
        if self.threat_response:
            threat_image = self.craft.sector.sensor_manager.target(threat, craft)
            self.threat_response.add_threat(threat_image)
            return
        assert(isinstance(self.craft, core.Ship))
        self.threat_response = combat.FleeOrder.create_flee_order(self.craft, self.gamestate)
        self.threat_response.observe(self)
        threat_image = self.craft.sector.sensor_manager.target(threat, self.craft)
        self.threat_response.add_threat(threat_image)
        assert(not self.threat_response.is_complete())
        self.craft.prepend_order(self.threat_response)

