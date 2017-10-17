#pragma once

#include "FockOperator.h"

class HartreeFock : public FockOperator {
public:
    HartreeFock(KineticOperator &t,
                NuclearPotential &v,
                CoulombOperator &j,
                ExchangeOperator &k,
                ReactionPotential *u = 0)
            : FockOperator(&t, &v, &j, &k, 0, u) { }
    virtual ~HartreeFock() { }
};

