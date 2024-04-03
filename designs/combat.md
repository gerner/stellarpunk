# Combat

Missiles and point defense style projectile weapons.
TBD sensors, but basically you're hard to detect. Using thrust makes you a lot
easier to detect.

## Sensors

Ships have a sensor profile. This represents how easily the ship can be
detected. This is influenced by size, mass, if the ship is thrusting, if the
ship has active sensors, if the ship has an active transponder.

The ease of detecting a craft is also influenced by the detecting ship's
sensor power, distance between target and detector and sector conditions.

A ship's transponder transmits precise information about location, heading,
velocity, and identification. The transponder is designed to make it easy to 
detect exactly where a ship is in almost any situation.

There's social and legal pressure to use an accurate transponder in civilized
areas. Not using a transponder is a serious offense. Spoofing transponders
is... impossible? TBD.

Ships have active and passive sensors. Active sensors are very powerful and
make it easy to see many targets. But they dramatically increase a ship's
sensor profile. Generally, if a ship uses active sensors, it's easy to detect
with passive sensors. Craft with passive sensors are generally difficult to
detect with active sensors.

Passive sesnsors are much less effective. They can detect some objects very
easily (e.g. ships with active transponders, massive objects like asteroids,
etc.). In some cases craft with active sensors might be hard to detect with
passive sensors. But generally active sensors give away position information to
passive sensors. Passive sensors have a very hard time detecting other craft
with passive sensors.

|             | Active    | Passive   |
|-------------|-----------|-----------|
| Transponder | Trivial   | Trivial+  |
| Active      | Easy      | Moderate- |
| Passive     | Moderate+ | Difficult |


### Algorithm

$p_{target} = \frac{(p_{base} + c_F * F + c_s * s + c_i * i) * w}{(c_d * d^2)}$

$p_{base} = c_m * m + c_r * r$

Where:
 * $p_{target}$ is the target sensor profile, after other factors are considered
 * $p_{base}$ is the base sensor profile depending on constant ship factors
 * $m$ is the mass of the ship
 * $r$ is the radius of the ship
 * $F$ is the thrust force the ship is applying and $c_t$ is a tuning coefficient associated with thrust in general
 * $s$ is the sensor power (passive or active) and $c_s$ is a tuning coefficient associated with sensor power
 * $i$ is the transponder factor (0 if off, 1 is on) and $c_i$ is a tuning coefficient
 * $w$ is the sector weather
 * $d$ is the distance between target and detector and $c_d$ is a tuning coefficient associated with distance in general

$q = c_q / ({c_{s_{detector]} * s_{detector}+1)$

Where:
 * $q$ is the sensor threshold of the detector
 * $c_q$ is a coefficient associated with sensor threshold
 * $s_{detector}$ is the sensor power of the detector
 * $c_{s_{detector}}$ is a tuning coefficient associated with detector power

A ship is detected if

$p_{target} > q$

That is, a ship is detected if it's target profile exceeds the sensor
threshold of the detector.

### Tuning

Note that collision detection looks for collision threats in a disk of radius
between 500m and 10km and a triangle forward of height up to 45km and base up
to 10km. Typically it's much smaller than that.

Sectors are typically 1-4Mm across. Sensors are tuned so that a passive ship
can just detect a passive, stationary ship at 100km. So 100km is typically
always visible. A passive ship can just detect an active, stationary ship at
1000km. So active ships can generally always be detected halfway across the
sector. This might take several minutes to cover. In that time the target might
have moved.

A passive moving craft has 10x the profile of a passive stationary one. An
active craft has 100x the profile of a passive stationary one. Turning on the
transponder increases the profile 1000x over a passive stationary craft and can
almost always be seen in the sector.

