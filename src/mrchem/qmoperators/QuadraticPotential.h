#ifndef QUADRATICPOTENTIAL_H
#define QUADRATICPOTENTIAL_H

#include "NuclearPotential.h"

class QuadraticPotential : public NuclearPotential {
public:
    QuadraticPotential(double Z, const double *R = 0, double S = 1.0e-7) : NuclearPotential(Z, R, S) { }
    virtual ~QuadraticPotential() { }

    virtual void setup(double prec) {
        if (IS_EQUAL(prec, this->apply_prec)) return;

        setApplyPrec(prec);
        if (this->hasReal()) MSG_ERROR("Potential not properly cleared");
        if (this->hasImag()) MSG_ERROR("Potential not properly cleared");

        MWProjector<3> project(this->apply_prec, this->max_scale);

        NuclearFunction &nuc_func = this->func;
        auto f = [nuc_func] (const double *r) -> double {
            double f_r = nuc_func.evalf(r);
            return f_r*f_r;
        };

        Timer timer;
        this->allocReal();
        project(this->real(), f);
        timer.stop();

        int n = this->getNNodes();
        double t = timer.getWallTime();
        TelePrompter::printTree(0, "Quadratic potential", n, t);
    }
};

#endif // QUADRATICPOTENTIAL_H
