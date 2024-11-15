# Combat

Missiles and point defense style projectile weapons.
TBD sensors, but basically you're hard to detect. Using thrust makes you a lot
easier to detect.

## Sensors

see `sensors.md` for more info.

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

### Tuning Sensors

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

### Sensor Noise

Model noise as a bias in our sensor readings: location and velocity. The
magnitude of this bias increases with decreasing sensor fidelity. The bias also
drifts over time. We can model this by recomputing the bias every time a sensor
reading is updated and mixing the new reading with the old one weighting the
new bias more with increasing time between measurements.

```
coeff = -param / (x+param) + 1
effective_bias = coeff * new_bias + (1-coeff) old_bias
```


## Engaging Combat

If a craft launches an attack on a target, the target should respond
tactically. This means they should know when the attack has been launched, if
sensors allow.

A target might face many threats (e.g. several incoming missiles), and there
might be several attackers. The target needs to track all of these. And this
set of threats might change during the engagement.

## Fighting

Attackers want to:
 * keep confidence in sensor readings of target
   * have an fresh sensor reading of the target
   * search for the target if not fresh
 * get into standoff range appropriate for available/preferred weapons
 * launch attacks when ready, in range, confident in hitting target
 * stay hidden

Targets want to:
 * potentially shift into attacking (depending on ???)
 * stay hidden
 * get far away from attacker and threats (i.e. missiles)
 * have fresh-ish sensor reading of threats (especially inbound missiles)

## Defense

Primarily ships depend on stealth and evasion for defense. If a craft is
threatened, the best thing they can do is deactivate their transponder and
active sensors and flee. Failing that, a craft can depend on point defense
systems to shoot down incoming missiles.

### Point Defense

Point defense systems work by firing projectiles at an incoming threat to try
and destroy the threat and alter it's trajectory by kinetic energy. The idea is
to fire many projectiles rapidly along the incoming threat trajectory. Point
defense can be used offensively, but generally it operates as close range, a
few km to 10 or 20 km.

Point defense isn't perfect. It operates by predicting motion of a threat to
try and get a projectile to intercept the threat. This can be easy to defeat.
On the other hand PD systems have very high rates of fire with reasonably high
muzzle velocities, so they can fire more projectiles quickly to account for
threat course changes. Still, it's possible, inevitable that a threat will get
through eventually.

### Decoys

POTENTIALLY

Decoys try to confuse sensor systems, especially simple sensor systems like
those on missiles or drones or sensors operating at long range. Decoys work
best when a threat is reacquiring a target after losing sensor contact,
especially if the decoy is positioned close to the predicted location based on
last sensor contact. The longer a decoy is in sensor contact, the more likely
it is to fail to confuse a threat's sesnsors.

### Environmental Stealth

POTENTIALLY

Localized particle or debris fields help hide ships by reducing a ship's sensor
profile. The tradeoff is that sensor threshold (ability to detect other ships)
is also reduced. But for a fleeing ship a sensor impacting area can be a safe
destination within a sector. Once a ship enters the area it can travel
unpredictably making the attacker lose the fleeing ship.

### Running out of Fuel

POTENTIALLY

Running out of fuel can be a death sentence for a ship. Even if a defender's
destruction is inevitable, it's not generally worth it to sacrifice the
attacker's ship by spending all fuel hunting the target down. An effective
defense strategy is simply to run away and wait out the attacker. At some point
the attacker has to give up the hunt.

## Damage and Destruction

Crafts are fragile. A coolant leak, losing cabin pressure, electrical or
control system damage can all disable or destroy a craft. However, one-shot
kills are still moderately uncommon. Ships have systems that can take damage
and become disabled before the entire craft is inoperable.

Independent systems take damage or become disabled? Has impact on gameplay?
Ship or component hitpoints?
Based on location where damage occurs? (e.g. stern vs bow of the craft)

### Damage Control/Mechanics

Gameplay mechanics to repair or work around component damage?
Ship performance impacted by damage? (if so, it should make sense like
positional damage)

### Collateral Damage

What happens if a third party gets hit during a firefight? By AI? By player?
X handles this with temporary hostility and some interaction between that
temporary hostility with faction-wide relations.
