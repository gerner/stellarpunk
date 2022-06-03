import logging
import curses
from curses import textpad
from typing import Any

from stellarpunk import interface, util, core
from stellarpunk.interface import sector as sector_interface

class UniverseView(interface.View):
    def __init__(self, gamestate:core.Gamestate, *args:Any, **kwargs:Any):
        super().__init__(*args, **kwargs)

        self.gamestate = gamestate

        # position of the universe sector cursor (in universe-sector coords)
        self.ucursor_x = 0
        self.ucursor_y = 0
        self.sector_maxx = 0
        self.sector_maxy = 0

        self.in_sector = False

    @property
    def viewscreen(self) -> curses.window:
        return self.interface.viewscreen

    def initialize(self) -> None:
        self.logger.info(f'entering universe mode')
        self.pan_camera()
        self.interface.reinitialize_screen(name="Universe Map")

    def focus(self) -> None:
        super().focus()
        self.in_sector = False
        self.interface.reinitialize_screen(name="Universe Map")

    def pan_camera(self) -> None:
        view_y = self.ucursor_y*(interface.Settings.UMAP_SECTOR_HEIGHT+interface.Settings.UMAP_SECTOR_YSEP)
        view_x = self.ucursor_x*(interface.Settings.UMAP_SECTOR_WIDTH+interface.Settings.UMAP_SECTOR_XSEP)

        self.viewscreen.move(view_y+1, view_x+1)

        # pan the camera so the selected sector is always in view
        if view_x < self.interface.camera_x:
            self.interface.camera_x = view_x
        elif view_x > self.interface.camera_x + self.interface.viewscreen_width - interface.Settings.UMAP_SECTOR_WIDTH:
            self.interface.camera_x = view_x - self.interface.viewscreen_width + interface.Settings.UMAP_SECTOR_WIDTH
        if view_y < self.interface.camera_y:
            self.interface.camera_y = view_y
        elif view_y > self.interface.camera_y + self.interface.viewscreen_height - interface.Settings.UMAP_SECTOR_HEIGHT:
            self.interface.camera_y = view_y - self.interface.viewscreen_height + interface.Settings.UMAP_SECTOR_HEIGHT

    def move_ucursor(self, direction:int) -> None:
        old_x = self.ucursor_x
        old_y = self.ucursor_y

        if direction == ord('w'):
            self.ucursor_y -= 1
        elif direction == ord('a'):
            self.ucursor_x -= 1
        elif direction == ord('s'):
            self.ucursor_y += 1
        elif direction == ord('d'):
            self.ucursor_x += 1
        else:
            raise ValueError(f'unknown direction {direction}')

        if self.ucursor_x < 0:
            self.ucursor_x = 0
            self.interface.status_message("no more sectors to the left", curses.color_pair(1))
        elif self.ucursor_x > self.sector_maxx:
            self.ucursor_x = self.sector_maxx
            self.interface.status_message("no more sectors to the right", curses.color_pair(1))

        if self.ucursor_y < 0:
            self.ucursor_y = 0
            self.interface.status_message("no more sectors upward", curses.color_pair(1))
        elif self.ucursor_y > self.sector_maxy:
            self.ucursor_y = self.sector_maxy
            self.interface.status_message("no more sectors downward", curses.color_pair(1))

    def draw_umap_sector(self, y:int, x:int, sector:core.Sector) -> None:
        """ Draws a single sector to viewscreen starting at position (y,x) """

        textpad.rectangle(self.viewscreen, y, x, y+interface.Settings.UMAP_SECTOR_HEIGHT-1, x+interface.Settings.UMAP_SECTOR_WIDTH-1)

        if (self.ucursor_x, self.ucursor_y) == (sector.x, sector.y):
            self.viewscreen.addstr(y+1,x+1, sector.short_id(), curses.A_STANDOUT)
        else:
            self.viewscreen.addstr(y+1,x+1, sector.short_id())

        self.viewscreen.addstr(y+2,x+1, sector.name)

        for resource, asteroids in sector.asteroids.items():
            amount = sum(map(lambda x: x.cargo[x.resource], asteroids))
            icon = interface.Icons.ASTEROID
            icon_attr = curses.color_pair(interface.Icons.RESOURCE_COLORS[resource])
            self.viewscreen.addstr(y+3+resource, x+2, f'{icon} {amount:.2e}', icon_attr)

        self.viewscreen.addstr(y+interface.Settings.UMAP_SECTOR_HEIGHT-2, x+1, f'{len(sector.entities)} objects')

    def update_display(self) -> None:
        """ Draws a map of all sectors. """

        if self.in_sector:
            return

        self.viewscreen.erase()
        self.sector_maxx = -1
        self.sector_maxy = -1
        for (x,y), sector in self.gamestate.sectors.items():
            self.sector_maxx = max(self.sector_maxx, x)
            self.sector_maxy = max(self.sector_maxy, y)
            # claculate screen_y and screen_x from x,y
            screen_x = x*(interface.Settings.UMAP_SECTOR_WIDTH+interface.Settings.UMAP_SECTOR_XSEP)
            screen_y = y*(interface.Settings.UMAP_SECTOR_HEIGHT+interface.Settings.UMAP_SECTOR_YSEP)

            self.draw_umap_sector(screen_y, screen_x, sector)

        self.pan_camera()
        self.interface.refresh_viewscreen()

    def handle_input(self, key:int) -> bool:
        if key in (ord('w'), ord('a'), ord('s'), ord('d')):
            self.move_ucursor(key)
        elif key in (ord('\n'), ord('\r')):
            sector = self.gamestate.sectors[(self.ucursor_x, self.ucursor_y)]
            sector_view = sector_interface.SectorView(
                    sector, self.interface)
            self.interface.open_view(sector_view)
            # suspend input until we get focus again
            self.in_sector = True
        elif key == ord(":"):
            command_input = interface.CommandInput(self.interface)
            self.interface.open_view(command_input)

        return True

