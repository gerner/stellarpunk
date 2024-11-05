Sensors
=======

Much of this is redundant/related to what's in combat.md.

# Desired Behaviors

Low profile (wrt sensor threshold) objects should be unidentified. We don't
know the identity, position, velocity or even type of object in some cases.

We get better readings as a function of both the instantaneous properties of
the target and how long we've been targeting them. For example at first we
might only know that there's a sensor target somewhere in a region. We might
not even be able to exactly quantify that uncertainty or region. As the target
is reveals itself (e.g. gets closer, uses thrusters) or we get a better fix on
the target (because we've been observing it for some time) we get a better idea
of those properties.

In Cold Waters there's even a feature/minigame where you have to help the
system id a target by looking at a sonic spectrum waterfall of the target.


## Sensor Noise

Model noise as a bias in our sensor readings: location and velocity. The
magnitude of this bias increases with decreasing sensor fidelity. The bias also
drifts over time. We want to avoid large drifts in the bias (e.g. jitter).

As fidelity improves, the bias should generally drift toward the actual
position, reducing the magnitude of the bias on average. The opposite is true
for deteriorating fidelity.

A given sensor image should have approximately the same or better accuracy as
it evolves over time as a newly targeted image with the same current sensor
fideltiy. That is, the accuracy of a sensor image targeted at one point and
then drifts over some time to a state with some new fidelity should be as
accurate or better than a new image of the same target made now.

We can pick an initial bias related to a distance and angle offset from the
detector's point of view. This distance and angle should decrease with sensor
fidelity.

We can pick an initial bias based on the profile/threshold ratio. We can then
scale the magnitude of this bias (in terms of distance from true location) as
we get further sensor readings. With each updated reading we should scale the
magnitude toward a magnitude appropriate for that updated reading
(profile/threshold ratio). We can further add noise on top of the updated
reading.

```
coeff = -param / (x+param) + 1
effective_bias = coeff * new_bias + (1-coeff) old_bias
```


# Display for Player

Player only gets to see SensorImage objects, not the underlying entity, unless
strictly necessary or the player's ship has fully resolved the object.

### Issues/TODOs
* icons and other entity display concerns for sensor images
* various target properties generally needed which might not be accessible if a
  sensor image is copied.

# Questions/Scenarios

* What happens if transponder is on? In concept this is transmitting full
  telemetry about the target, so shouldn't this make the target visible
  everywhere?
* What happens if transponder gets turned off? In concept we had full
  telemetry, so shouldn't we know exactly where the target is immediately after
  transponder goes off?
* If sensor fidelity decreases and then increases, should we keep the high
  accuracy position information or should that accuracy drop immediately? Do we
  know what the fidelity is exactly?
* How does target identification work? Do we know what kind of target we have
  right away? Is that revealed over time? Are there multiple levels or even a
  continuum of target identity information as fidelity increases?
* Releated to target identification, if we are special casing the transponder
  case, dow we treat some sensor contacts that are very well resolved, but
  without a transpondersimilarly? e.g. an asteroid that we want the player to
  interact with the same certainty of information as they would a friendly craft
  or station.

