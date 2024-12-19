import tempfile

from stellarpunk import sim, core
from stellarpunk.serialization import util as s_util

def test_to_int():
    with tempfile.TemporaryFile() as fp:
        v1 = 42
        bytes_written = s_util.int_to_f(v1, fp)
        fp.flush()
        assert bytes_written == fp.tell()
        fp.seek(0)
        v2 = s_util.int_from_f(fp)
        assert v2 == v1
        assert bytes_written == fp.tell()

def test_save_load_registry(event_manager, generator):
    game_saver = sim.initialize_save_game(generator, event_manager)
    with tempfile.TemporaryFile() as fp:
        bytes_written = game_saver._save_registry(fp)
        fp.flush()
        assert bytes_written == fp.tell()

        fp.seek(0)
        game_saver._load_registry(fp)
        assert bytes_written == fp.tell()

def test_trivial_gamestate(event_manager, gamestate, generator, player):
    assert player == gamestate.player
    game_saver = sim.initialize_save_game(generator, event_manager)
    filename = game_saver.save(gamestate)
    g2 = game_saver.load(filename)
    #this won't work!
    #assert g2 == gamestate

    assert gamestate.random.integers(42) == g2.random.integers(42)
    assert gamestate.player.entity_id == g2.player.entity_id
    assert gamestate.production_chain == g2.production_chain

def test_event_state(event_manager, gamestate, generator):
    # trigger some events
    pass
