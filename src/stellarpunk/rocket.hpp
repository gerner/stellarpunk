#include <set>
#include <algorithm>
#include <cassert>
#include "chipmunk/chipmunk.h"

const double G_0 = 9.80665;
cpBody cpBodySingleton;

class RocketModel;

class RocketSpace {
    private:
        double timestamp_=0;
        std::set<RocketModel*> thrust_rocket_models_;
    public:
        double get_timestamp() const {
            return timestamp_;
        }
        void mark_thrust(RocketModel* rocket_model) {
            thrust_rocket_models_.insert(rocket_model);
        }
        void unmark_thrust(RocketModel* rocket_model) {
            thrust_rocket_models_.erase(rocket_model);
        }
        void tick(const double dt, const double current_time);
};

class RocketModel {
    private:
        // intrinsic properties
        RocketSpace* space_;
        cpBody* body_;
        double i_sp_;
        double max_thrust_;
        double max_fine_thrust_;
        double max_torque_;

        // variable state
        cpVect force_;
        double torque_;
        double propellant_;

    public:
    RocketModel() : RocketModel(NULL, NULL, 0.0, 0.0, 0.0, 0.0) { }
    RocketModel(RocketSpace* space, cpBody* body, double i_sp, double max_thrust, double max_fine_thrust, double max_torque) {
        space_ = space;
        body_ = body;
        force_ = cpvzero;
        propellant_ = 0;

        i_sp_ = i_sp;
        max_thrust_ = max_thrust;
        max_fine_thrust_ = max_fine_thrust;
        max_torque_ = max_torque;
    }

    ~RocketModel() {
        if (cpveql(force_, cpvzero)) {
            space_->unmark_thrust(this);
        }
    }

    void destroy() {
        if (cpveql(force_, cpvzero)) {
            space_->unmark_thrust(this);
        }
        force_ = cpvzero;
        body_ = NULL;
    }

    // models consuming propellant at the current thrust level for given
    // timestep
    // returns true iff that consumed all the propellant
    bool consume_propellant(const double dt) {
        assert(!cpveql(force_, cpvzero));
        assert(body_);
        double propellant_used = cpvlength(force_) / (i_sp_ * G_0) * dt;
        if (propellant_used > propellant_) {
            propellant_used = propellant_;
        }

        adjust_propellant(-propellant_used);

        //TODO: handle case where we run out of propellant this tick
        //  we should probably still allow some thrust and not just zero it out

        if (propellant_ == 0) {
            force_ = cpvzero;
            // we remove our thrust from the body, but leave any other external
            // forces in place
            body_->f = cpvsub(body_->f, force_);
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

    double get_max_thrust() const {
        return max_thrust_;
    }

    void set_max_thrust(const double max_thrust) {
        max_thrust_ = max_thrust;
    }

    double get_max_fine_thrust() const {
        return max_fine_thrust_;
    }

    void set_max_fine_thrust(double max_fine_thrust) {
        max_fine_thrust_ = max_fine_thrust;
    }

    double get_max_torque() const {
        return max_torque_;
    }

    void set_max_torque(const double max_torque) {
        max_torque_ = max_torque;
    }

    double get_propellant() const {
        return propellant_;
    }

    // sets the amount of propellant available
    // this does NOT update the physical properties of the physics body
    void set_propellant(const double propellant) {
        propellant_ = propellant;
    }

    cpVect get_force() const {
        return force_;
    }

    // sets force directly
    // this does NOT update the physical properties of the physics body
    void set_force(const cpVect force) {
        assert(cpvlength(force) <= max_thrust_+0.1);
        if (!cpveql(force, cpvzero) and cpveql(force_, cpvzero)) {
            space_->mark_thrust(this);
        } else if (cpveql(force, cpvzero) and !cpveql(force_, cpvzero)) {
            space_->unmark_thrust(this);
        }
        force_ = force;
    }

    double get_torque() const {
        return torque_;
    }

    // sets torque directly
    // this does NOT update the physical properties of the physics body
    void set_torque(const double torque) {
        assert(abs(torque) <= max_torque_+0.1);
        torque_ = torque;
    }

    void apply_force(const cpVect force) {
        assert(cpvlength(force) <= max_thrust_+0.1);
        assert(body_);

        //TODO: handle and somehow expose case where we have zero propellant
        if (cpveql(force, force_)) {
            return;
        }

        // book keeping to keep track of rocket models with non-zero thrust
        if (!cpveql(force, cpvzero) and cpveql(force_, cpvzero)) {
            space_->mark_thrust(this);
        } else if (cpveql(force, cpvzero) and !cpveql(force_, cpvzero)) {
            space_->unmark_thrust(this);
        }

        // remove current force and add the new force
        body_->f = cpvadd(cpvsub(body_->f, force_), force);

        force_ = force;
    }

    void apply_torque(const double torque) {
        assert(abs(torque) <= max_torque_+0.1);
        assert(body_);
        if (torque == torque_) {
            return;
        }

        // remove current torque and add the new torque
        body_->t += torque - torque_;
        torque_ = torque;
    }

    // updates amount of propellant available to given amount
    // this also updates the physical properties of the underlying physics body
    void adjust_propellant(const double delta) {
        // don't want to go negative
        assert(-delta <= propellant_);
        assert(body_);
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

};

void RocketSpace::tick(const double dt, const double current_time) {
    for(auto it = thrust_rocket_models_.begin(); it != thrust_rocket_models_.end();) {
        if((*it)->consume_propellant(dt)) {
            it = thrust_rocket_models_.erase(it);
        } else {
            it++;
        }
    }
    timestamp_ = current_time;
}

