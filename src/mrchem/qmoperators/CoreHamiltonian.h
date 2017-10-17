#pragma once

#include "FockOperator.h"

class CoreHamiltonian : public FockOperator {
public:
    CoreHamiltonian(KineticOperator &t,
                    NuclearPotential &v,
                    ReactionPotential *u = 0)
        : FockOperator(&t, &v, 0, 0, 0, u) { }
    virtual ~CoreHamiltonian() { }
};

