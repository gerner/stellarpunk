""" Tests for station production. """

import numpy as np

def test_basic_produce(gamestate, generator, sector, simulator):
    station = generator.spawn_station(sector, 0., 0., resource=gamestate.production_chain.ranks[0])

    resources_needed = gamestate.production_chain.adj_matrix[:,station.resource] * gamestate.production_chain.batch_sizes[station.resource]
    station.cargo += resources_needed * 3

    simulator.produce_at_station(station)

    next_batch_time = station.next_batch_time
    assert next_batch_time == gamestate.timestamp + gamestate.production_chain.production_times[station.resource]
    assert np.all(station.cargo == resources_needed * 2)

    gamestate.timestamp = station.next_batch_time/2
    simulator.produce_at_station(station)

    #make sure we don't produce yet
    assert station.cargo[station.resource] == 0.
    assert station.next_batch_time == next_batch_time
    gamestate.timestamp = station.next_batch_time
    simulator.produce_at_station(station)
    #TODO: make sure we produced once
    assert station.next_batch_time == 0.
    assert station.cargo[station.resource] == gamestate.production_chain.batch_sizes[station.resource]

def test_not_enough_resources(gamestate, generator, sector, simulator):
    station = generator.spawn_station(sector, 0., 0., resource=gamestate.production_chain.ranks[0])

    simulator.produce_at_station(station)
    assert station.next_batch_time == 0.
    assert np.all(station.cargo == 0.)

    gamestate.timestamp = 2.
    simulator.produce_at_station(station)
    assert station.next_batch_time == 0.
    assert np.all(station.cargo == 0.)
