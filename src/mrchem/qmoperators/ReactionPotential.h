#pragma once

#include "QMPotential.h"

class ReactionPotential : public QMPotential {
public:
    ReactionPotential();
    virtual ~ReactionPotential();

    void setup(double prec);
    void clear();
};

