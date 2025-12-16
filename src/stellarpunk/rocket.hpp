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

    bool consume_propellant(const double dt) {
        assert(thrust_ > 0);
        double propellant_used = thrust_ / (i_sp_ * G_0) * dt;
        //TODO: handle dropping mass (and moment) from body_
        //TODO: handle case where we go over
        if (propellant_used > propellant_) {
            propellant_ = 0;
            thrust_ = 0;
            body_->f = cpvzero;
            return true;
        }
        propellant_ -= propellant_used;
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

    void set_propellant(const double propellant) {
        propellant_ = propellant;
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

void rocket_tick(double dt) {
    for(auto it = thrust_rocket_models.begin(); it != thrust_rocket_models.end();) {
        //TODO: handle case where we've used up all the propellant
        if((*it)->consume_propellant(dt)) {
            it = thrust_rocket_models.erase(it);
        } else {
            it++;
        }
    }
}

