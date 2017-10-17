#pragma once

#include "FockOperator.h"

class Hartree : public FockOperator {
public:
    Hartree(KineticOperator &t,
            NuclearPotential &v,
            CoulombOperator &j,
            ReactionPotential *u = 0)
        : FockOperator(&t, &v, &j, 0, 0, u) { }
    virtual ~Hartree() { }
};

