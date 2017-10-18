#pragma once

#include "QMPotential.h"
#include "Nucleus.h"

template<int D> class DerivativeOperator;
class PoissonOperator;
class CavityFunction;

class ReactionPotential : public QMPotential {
public:
    ReactionPotential(double prec,
                      PoissonOperator &P,
                      DerivativeOperator<3> &D,
                      CavityFunction &cav,
                      Nuclei &nucs,
                      OrbitalVector &phi);
    virtual ~ReactionPotential();

    void setup(double prec);
    void clear();
    FunctionTree<3> &getNuclearDensity() {return *this->rho_nuc;}
protected:
    OrbitalVector *orbitals;
    PoissonOperator *poisson;
    DerivativeOperator<3> *derivative;
    CavityFunction *cavity;

    FunctionTree<3> *eps;
    FunctionTree<3> *eps_inv;
    FunctionTree<3> *U_n;
    FunctionTree<3> *d_eps[3];
    FunctionTree<3> *rho_nuc;
    FunctionTree<3> *rho_el;

    void calcElectronDensity(double prec);
    void calcNuclearDensity(double prec, const Nuclei &nucs);
    void calcDielectricFunction(double prec, const Nuclei &nucs);
};

