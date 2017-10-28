#include "ReactionPotential.h"
#include "CavityFunction.h"
#include "OrbitalVector.h"
#include "MWProjector.h"
#include "MWDerivative.h"
#include "MWConvolution.h"
#include "MWAdder.h"
#include "MWMultiplier.h"
#include "GridGenerator.h"
#include "PoissonOperator.h"
#include "DerivativeOperator.h"
#include "GaussFunc.h"
#include "Orbital.h"
#include "KAIN.h"

extern MultiResolutionAnalysis<3> *MRA;

using namespace std;

ReactionPotential::ReactionPotential(int hist,
                                     double prec,
                                     PoissonOperator &P,
                                     DerivativeOperator<3> &D,
                                     CavityFunction &cav,
                                     Nuclei &nucs,
                                     OrbitalVector &phi)
        : history(hist),
          orbitals(&phi),
          poisson(&P),
          derivative(&D),
          cavity(&cav) {

    calcNuclearDensity(prec, nucs);
    calcDielectricFunction(prec);
    calcSpilloverFunction(prec);
    this->U_n = new FunctionTree<3>(*MRA);
    this->U_n->setZero();
}

ReactionPotential::~ReactionPotential() {
    delete this->rho_nuc;
    delete this->eps;
    delete this->eps_inv;
    delete this->d_eps[0];
    delete this->d_eps[1];
    delete this->d_eps[2];
    delete this->U_n;
}




void ReactionPotential::calcElectronDensity(double prec) {
    Timer timer;
    MWAdder<3> add(prec, this->max_scale);
    MWMultiplier<3> mult(prec, this->max_scale);

    FunctionTreeVector<3> rho_vec;
    for (int i = 0; i < this->orbitals->size(); i++){
        Orbital &phi = this->orbitals->getOrbital(i);
        if (phi.hasReal()) {
            FunctionTree<3> *temp = new FunctionTree<3>(*MRA);
            mult(*temp, 1.0, phi.real(), phi.real());
            rho_vec.push_back(phi.getOccupancy(), temp);
        }
        if (phi.hasImag()) {
            FunctionTree<3> *temp = new FunctionTree<3>(*MRA);
            mult(*temp, 1.0, phi.imag(), phi.imag());
            rho_vec.push_back(-phi.getOccupancy(), temp);
        }
    }

    this->rho_el = new FunctionTree<3>(*MRA);
    add(*this->rho_el, rho_vec);
    rho_vec.clear(true);

    timer.stop();
    int n = this->rho_el->getNNodes();
    double t = timer.getWallTime();
    TelePrompter::printTree(0, "Electron density", n, t);
}

void ReactionPotential::calcNuclearDensity(double prec, const Nuclei &nucs) {
    Timer timer;
    TelePrompter::printHeader(0, "Projecting nuclear density");

    this->rho_nuc = new FunctionTree<3>(*MRA);

    GaussExp<3> gexp;
    double beta = pow(10.0, 6.0);
    double alpha = pow(beta/pi, 3.0/2.0);
    for (int i = 0; i < nucs.size(); i++) {
        double Z = -nucs[i].getCharge();
        GaussFunc<3> func(beta, Z*alpha, nucs[i].getCoord());
        gexp.append(func);
    }

    MWProjector<3> project(prec, this->max_scale);
    GridGenerator<3> grid(this->max_scale);
    grid(*this->rho_nuc, gexp);
    project(*this->rho_nuc, gexp);

    timer.stop();
    TelePrompter::printFooter(0, timer, 2);
}

void ReactionPotential::calcDielectricFunction(double prec) {
    Timer timer;
    TelePrompter::printHeader(0, "Projecting dielectric function");

    MWProjector<3> project(prec, this->max_scale);
    MWDerivative<3> apply(this->max_scale);

    this->cavity->setInverse(false);
    this->eps = new FunctionTree<3>(*MRA);
    project(*this->eps, *this->cavity);

    this->cavity->setInverse(true);
    this->eps_inv = new FunctionTree<3>(*MRA);
    project(*this->eps_inv, *this->cavity);

    for (int i = 0; i < 3; i++) {
        this->d_eps[i] = new FunctionTree<3>(*MRA);
        apply(*this->d_eps[i], *this->derivative, *this->eps, i);
    }

    timer.stop();
    TelePrompter::printFooter(0, timer, 2);
}


void ReactionPotential::calcSpilloverFunction(double prec) {
    Timer timer;
    TelePrompter::printHeader(0, "Projecting spillover function");

    CavityFunction &cav = *this->cavity;
    cav.setInverse(false);
    auto f = [cav] (const double *r) -> double {
        double cav_r = cav.evalf(r);
        double eps_0 = cav.getEpsilon_0();
        double eps_inf = cav.getEpsilon_inf();
        double denom = (eps_inf - eps_0);
        double result = 1.0;
        if (fabs(denom) > MachineZero) result = (cav_r - eps_0)/denom;
        return result;
    };

    MWProjector<3> project(prec, this->max_scale);
    this->eps_spill = new FunctionTree<3>(*MRA);
    project(*this->eps_spill, f);

    timer.stop();
    TelePrompter::printFooter(0, timer, 2);
}

void ReactionPotential::setup(double prec) {
    setApplyPrec(prec);

    MWAdder<3> add(this->apply_prec, this->max_scale);
    MWMultiplier<3> mult(this->apply_prec, this->max_scale);
    MWConvolution<3> apply(this->apply_prec, this->max_scale);
    MWDerivative<3> diff(this->max_scale);

    calcElectronDensity(prec);
    FunctionTree<3> rho(*MRA);
    add(rho, 1.0, *this->rho_el, 1.0, *this->rho_nuc);

    this->spill_out = this->eps_spill->dot(rho);
    this->spill_in = rho.integrate() - this->spill_out;

    //creating rho_eff (rho/eps)
    FunctionTree<3> rho_eff(*MRA);
    mult(rho_eff, 1.0, rho, *this->eps_inv);

    FunctionTree<3> *U_np1 = 0;
    FunctionTree<3> *U_r = 0;

    KAIN kain(this->history);

    int cycle = 0;
    bool converged = false;
    double errorU = 1.0;
    while (not converged) {
        cycle++;

        //derivative of electrostatic potential
        FunctionTree<3> dx_U(*MRA);
        FunctionTree<3> dy_U(*MRA);
        FunctionTree<3> dz_U(*MRA);

        diff(dx_U, *this->derivative, *this->U_n, 0);
        diff(dy_U, *this->derivative, *this->U_n, 1);
        diff(dz_U, *this->derivative, *this->U_n, 2);

        //creating gamma (grad_eps*grad_V)/(4pi*eps)
        FunctionTreeVector<3> gradeps_gradU;
        FunctionTree<3> dx_eps_dx_U(*MRA);
        FunctionTree<3> dy_eps_dy_U(*MRA);
        FunctionTree<3> dz_eps_dz_U(*MRA);

        mult(dx_eps_dx_U, 1.0, *this->d_eps[0], dx_U);
        mult(dy_eps_dy_U, 1.0, *this->d_eps[1], dy_U);
        mult(dz_eps_dz_U, 1.0, *this->d_eps[2], dz_U);

        gradeps_gradU.push_back(1.0, &dx_eps_dx_U);
        gradeps_gradU.push_back(1.0, &dy_eps_dy_U);
        gradeps_gradU.push_back(1.0, &dz_eps_dz_U);

        FunctionTree<3> temp_func(*MRA);
        add(temp_func, gradeps_gradU);
        temp_func *= 1.0/(4.0*pi);
        gradeps_gradU.clear();

        FunctionTree<3> gamma(*MRA);
        mult(gamma, 1.0, temp_func, *this->eps_inv);

        //calculating V_tot
        FunctionTree<3> sum_rhoeff_gamma(*MRA);
        add(sum_rhoeff_gamma, 1.0, rho_eff, 1.0, gamma);

        //applying Poisson function
        U_np1 = new FunctionTree<3>(*MRA);
        apply(*U_np1, *this->poisson, sum_rhoeff_gamma);

        //preparing for next iteration
        FunctionTree<3> *dU_n = new FunctionTree<3>(*MRA);
        add(*dU_n, 1.0, *U_np1, -1.0, *U_n);

        if (cycle > 1 and this->history > 0) {
            OrbitalVector phi_n(0);
            OrbitalVector dPhi_n(0);
            phi_n.push_back(1, 2, Orbital::Paired);
            dPhi_n.push_back(1, 2, Orbital::Paired);

            phi_n.getOrbital(0).setReal(U_n);
            dPhi_n.getOrbital(0).setReal(dU_n);

            double plevel = TelePrompter::setPrintLevel(-1);
            kain.accelerate(prec, phi_n, dPhi_n);
            TelePrompter::setPrintLevel(plevel);

            U_n = phi_n.getOrbital(0).getReal();
            dU_n = dPhi_n.getOrbital(0).getReal();

            phi_n.clear(false);
            dPhi_n.clear(false);
        }

        errorU = sqrt(dU_n->getSquareNorm()/U_np1->getSquareNorm());
        println(0, setw(3) << cycle << ":  " << errorU);

        delete U_np1;
        U_np1 = new FunctionTree<3>(*MRA);
        add(*U_np1, 1.0, *U_n, 1.0, *dU_n);
        delete dU_n;

        if (errorU < prec) {
            converged = true;

            //calculating V_r
            FunctionTree<3> sum_rhoeff_gamma_neg_rho(*MRA);
            add(sum_rhoeff_gamma_neg_rho, 1.0, sum_rhoeff_gamma, -1.0, rho);

            //applying Poisson operator
            if (U_r != 0) delete U_r;
            U_r = new FunctionTree<3>(*MRA);
            apply(*U_r, *this->poisson, sum_rhoeff_gamma_neg_rho);
        }

        if (U_n != 0) delete U_n;
        U_n = U_np1;
    }
    this->re = U_r;
    delete this->rho_el;
}

void ReactionPotential::clear() {
    clearReal(true);
    clearImag(true);
    clearApplyPrec();
}
