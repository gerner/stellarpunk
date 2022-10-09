# Trading Integration

Integrating work from economy simulation to enable trading and price setting.

# Elements

## Characters

My plan is to implement a skeleton character system at the same time.
Characters will own assets like ships and stations. Money and price setting
will sit with characters. Imagine getting an economy window where you can get a
balance sheet and other economy data for a station and set prices of goods and
the budget to spend on inputs.

Each character will have AI behaviors attached to it. In the long run we'll
have motivations and agenda for each character. For now we'll just focus on
economics, production, trading, managing a balance and trying to profit
maximize. The player will control one of these characters.

Depending on the assets a character controls, they will have some
responsibilities that lead to gameplay behaviors. If you own a ship you can
make decisions about trading, like where to go mine or buy/sell goods. This
turns into ship activities, like movement or cargo transfers. If you own a
station you need to manage the input/output prices and setting a budget.

Punt on character relationships/hierarchies right now.

## Economy

We'll rely on the logic developed in the economic simulation to handle how
trading works from an economic/financial perspective. Characters will set input
and output prices and a budget, or amount desired. The AI will base prices on a
min/max scheme based on historical input/output costs, along with some
optimization of revenue/costs. Budgets get based on targeting a desired level
of inventory to support production over a given time period.

Trading can be based on straight forward revenue/time maximization, taking into
account prices, volumes and time to travel.

* Econ sim for history building and universe generation?
* Continue with matrix based data storage/manipulation during gameplay?

P0 goal is that economy can function under favorable conditions. After that
comes reasonable profit maximization. After that comes more complex behaviors
modelling responsiveness to other characters and market conditions.

# Plan

- [ ] character framework: balance, ownership, decision making hooks
- [ ] player controlled character
- [ ] booting up the economy/initial asset/character/settings
- [ ] miners choosing asteroids to mine and station to sell at
- [ ] traders choosing good and stations to buy/sell at
- [ ] station owners choosing prices/budget
- [ ] testing and tuning on economy
