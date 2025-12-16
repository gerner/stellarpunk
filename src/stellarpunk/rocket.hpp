#include <vector>
#include <algorithm>
#include <cassert>
#include "chipmunk/chipmunk.h"

const double G_0 = 9.80665;
cpBody cpBodySingleton;

class RocketModel;

std::vector<RocketModel*> thrust_rocket_models;

class RocketModel {
    private:
        double thrust_;
        double propellant_;
        double i_sp_;
        cpBody* body_;

    public:
    RocketModel() : RocketModel(& cpBodySingleton) { }

    RocketModel(cpBody* body) {
        body_ = body;
        thrust_ = 0;
        propellant_ = 0;
        i_sp_ = 452.3;
    }

    ~RocketModel() {
        if (thrust_ > 0) {
            thrust_rocket_models.erase(
                std::remove(thrust_rocket_models.begin(), thrust_rocket_models.end(), this),
                thrust_rocket_models.end());
        }
    }

    bool consume_propellant(const double dt) {
        assert(thrust_ > 0);
        double propellant_used = thrust_ / (i_sp_ * G_0) * dt;
        //TODO: handle case where we go over
        /*if (propellant_used > propellant_) {
            propellant_ = 0;
            thrust_ = 0;
            body_->f = cpvzero;
            return true;
        }*/
        propellant_ -= propellant_used;
        return false;
    }

    double get_propellant() const {
        return propellant_;
    }

    void set_thrust(double thrust) {
        //TODO: handle and somehow expose case where we have zero propellant
        if (thrust == thrust_) {
            return;
        }

        // book keeping to keep track of rocket models with non-zero thrust
        if (thrust > 0. and thrust_ == 0) {
            thrust_rocket_models.push_back(this);
        } else if (thrust == 0. and thrust_ > 0) {
            thrust_rocket_models.erase(
                std::remove(thrust_rocket_models.begin(), thrust_rocket_models.end(), this),
                thrust_rocket_models.end());
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

