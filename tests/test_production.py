""" Tests for station production. """

import numpy as np

from stellarpunk import econ, agenda

def test_basic_produce(gamestate, generator, sector, simulator):
    station = generator.spawn_station(sector, 0., 0., resource=gamestate.production_chain.ranks[0])

    producer_owner = generator.spawn_character(station)
    producer_owner.take_ownership(station)
    producer_agendum = agenda.StationManager.create_station_manager(station, producer_owner, gamestate)
    producer_owner.add_agendum(producer_agendum)

    resources_needed = gamestate.production_chain.adj_matrix[:,station.resource] * gamestate.production_chain.batch_sizes[station.resource]
    station.cargo += resources_needed * 3

    # tick once to get production set up
    timestamp_at_tick = 1.
    gamestate.timestamp = timestamp_at_tick
    simulator.tick(simulator.dt)

    next_batch_time = station.next_batch_time
    assert next_batch_time == timestamp_at_tick + gamestate.production_chain.production_times[station.resource]
    assert np.all(station.cargo == resources_needed * 2)

    gamestate.timestamp = station.next_batch_time/2
    simulator.tick(simulator.dt)

    #make sure we don't produce yet
    assert producer_agendum.produced_batches == 0
    assert station.cargo[station.resource] == 0.
    assert station.next_batch_time == next_batch_time

    gamestate.timestamp = station.next_batch_time + 1.0
    simulator.tick(simulator.dt)
    assert producer_agendum.produced_batches == 1
    assert station.next_batch_time == 0.
    assert station.cargo[station.resource] == gamestate.production_chain.batch_sizes[station.resource]

def test_not_enough_resources(gamestate, generator, sector, simulator):
    station = generator.spawn_station(sector, 0., 0., resource=gamestate.production_chain.ranks[0])

    producer_owner = generator.spawn_character(station)
    producer_owner.take_ownership(station)
    producer_agendum = agenda.StationManager.create_station_manager(station, producer_owner, gamestate)
    producer_owner.add_agendum(producer_agendum)

    gamestate.timestamp = 1.
    simulator.tick(simulator.dt)
    assert station.next_batch_time == 0.
    assert np.all(station.cargo == 0.)

    gamestate.timestamp = 2.
    simulator.tick(simulator.dt)
    assert station.next_batch_time == 0.
    assert np.all(station.cargo == 0.)
