#include <set>
#include <algorithm>
#include <cassert>
#include "chipmunk/chipmunk.h"

const double G_0 = 9.80665;
cpBody cpBodySingleton;

class RocketModel;

std::set<RocketModel*> thrust_rocket_models;

class RocketModel {
    private:
        double thrust_;
        double propellant_;
        double i_sp_;
        cpBody* body_;

    public:
    RocketModel() : RocketModel(&cpBodySingleton, 0.0) { }

    RocketModel(cpBody* body, double i_sp) {
        body_ = body;
        thrust_ = 0;
        propellant_ = 0;
        i_sp_ = i_sp;
    }

    ~RocketModel() {
        if (thrust_ > 0) {
            thrust_rocket_models.erase(this);
        }
    }

    // models consuming propellant at the current thrust level for given
    // timestep
    // returns true iff that consumed all the propellant
    bool consume_propellant(const double dt) {
        assert(thrust_ > 0);
        double propellant_used = thrust_ / (i_sp_ * G_0) * dt;
        if (propellant_used > propellant_) {
            propellant_used = propellant_;
        }

        adjust_propellant(-propellant_used);

        if (propellant_ == 0) {
            thrust_ = 0;
            body_->f = cpvzero;
            return true;
        }
        return false;
    }

    double get_i_sp() const {
        return i_sp_;
    }

    void set_i_sp(const double i_sp) {
        i_sp_ = i_sp;
    }

    double get_propellant() const {
        return propellant_;
    }

    // sets the amount of propellant available
    // this does NOT update the physical properties of the physics body
    void set_propellant(const double propellant) {
        propellant_ = propellant;
    }

    // updates amount of propellant available to given amount
    // this also updates the physical properties of the underlying physics body
    void adjust_propellant(const double delta) {
        // don't want to go negative
        assert(-delta <= propellant_);
        if (delta == 0) {
            return;
        }
        propellant_ += delta;

        double new_mass = body_->m + delta;
        double ratio = new_mass / body_->m;
        double new_moment = body_->i * ratio;

        cpBodySetMass(body_, new_mass);
        cpBodySetMoment(body_, new_moment);
    }

    double get_thrust() const {
        return thrust_;
    }

    void set_thrust(const double thrust) {
        //TODO: handle and somehow expose case where we have zero propellant
        if (thrust == thrust_) {
            return;
        }

        // book keeping to keep track of rocket models with non-zero thrust
        if (thrust > 0. and thrust_ == 0) {
            thrust_rocket_models.insert(this);
        } else if (thrust == 0. and thrust_ > 0) {
            thrust_rocket_models.erase(this);
        }

        thrust_ = thrust;
    }
};

void rocket_tick(const double dt) {
    for(auto it = thrust_rocket_models.begin(); it != thrust_rocket_models.end();) {
        if((*it)->consume_propellant(dt)) {
            it = thrust_rocket_models.erase(it);
        } else {
            it++;
        }
    }
}

