#pragma once

#include "FockOperator.h"

class DFT : public FockOperator {
public:
    DFT(KineticOperator &t,
        NuclearPotential &v,
        CoulombOperator &j,
        XCOperator &xc,
        ExchangeOperator *k = 0,
        ReactionPotential *u = 0)
        : FockOperator(&t, &v, &j, k, &xc, u) { }
    virtual ~DFT() { }
};

