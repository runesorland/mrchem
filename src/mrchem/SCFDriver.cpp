#include <fstream>

#include "eigen_disable_warnings.h"
#include "mrchem.h"
#include "SCFDriver.h"
#include "TelePrompter.h"
#include "MathUtils.h"

#include "InterpolatingBasis.h"
#include "MultiResolutionAnalysis.h"

#include "CoreHamiltonian.h"
#include "Hartree.h"
#include "HartreeFock.h"
#include "DFT.h"

#include "HelmholtzOperatorSet.h"
#include "OrbitalOptimizer.h"
#include "EnergyOptimizer.h"
#include "LinearResponseSolver.h"
#include "KAIN.h"

#include "Molecule.h"
#include "OrbitalVector.h"
#include "OrbitalProjector.h"
#include "DensityProjector.h"

#include "SCFEnergy.h"
#include "DipoleMoment.h"
#include "Magnetizability.h"
#include "NMRShielding.h"
#include "SpinSpinCoupling.h"
#include "HyperFineCoupling.h"

#include "PoissonOperator.h"
#include "ABGVOperator.h"
#include "PHOperator.h"

#include "H_E_dip.h"
#include "H_B_dip.h"
#include "H_BB_dia.h"
#include "H_BM_dia.h"
#include "H_M_pso.h"
#include "H_M_fc.h"
#include "SpinOperator.h"
#include "KineticOperator.h"
#include "NuclearPotential.h"
#include "CoulombPotential.h"
#include "ExchangePotential.h"
#include "ReactionPotential.h"
#include "XCPotential.h"
#include "XCFunctional.h"
#include "IdentityOperator.h"

#include "CavityFunction.h"

using namespace std;
using namespace Eigen;

extern MultiResolutionAnalysis<3> *MRA; // Global MRA

SCFDriver::SCFDriver(Getkw &input) {
    max_scale = MRA->getMaxScale();
    rel_prec = input.get<double>("rel_prec");
    nuc_prec = input.get<double>("nuc_prec");

    gauge = input.getDblVec("MRA.gauge_origin");
    center_of_mass = input.get<bool>("MRA.center_of_mass");

    sol_slope = input.get<double>("Solvent.slope");
    sol_eps_0 = input.get<double>("Solvent.eps_0");
    sol_eps_inf = input.get<double>("Solvent.eps_inf");
    sol_alpha = input.get<double>("Solvent.alpha");
    sol_kain = input.get<int>("Solvent.kain");

    diff_kin = input.get<string>("Derivatives.kinetic");
    diff_orb = input.get<string>("Derivatives.h_orb");
    diff_pso = input.get<string>("Derivatives.h_pso");

    calc_scf_energy = input.get<bool>("Properties.scf_energy");
    calc_dipole_moment = input.get<bool>("Properties.dipole_moment");
    calc_quadrupole_moment = input.get<bool>("Properties.quadrupole_moment");
    calc_magnetizability = input.get<bool>("Properties.magnetizability");
    calc_nmr_shielding = input.get<bool>("Properties.nmr_shielding");
    calc_hyperfine_coupling = input.get<bool>("Properties.hyperfine_coupling");
    calc_spin_spin_coupling = input.get<bool>("Properties.spin_spin_coupling");
    calc_polarizability = input.get<bool>("Properties.polarizability");
    calc_hyperpolarizability = input.get<bool>("Properties.hyperpolarizability");
    calc_optical_rotation = input.get<bool>("Properties.optical_rotation");

    nmr_perturbation = input.get<string>("NMRShielding.perturbation");
    nmr_nucleus_k = input.getIntVec("NMRShielding.nucleus_k");
    hfcc_nucleus_k = input.getIntVec("HyperfineCoupling.nucleus_k");
    sscc_nucleus_k = input.getIntVec("SpinSpinCoupling.nucleus_k");
    sscc_nucleus_l = input.getIntVec("SpinSpinCoupling.nucleus_l");
    pol_velocity = input.get<bool>("Polarizability.velocity");
    pol_frequency = input.getDblVec("Polarizability.frequency");
    optrot_velocity = input.get<bool>("OpticalRotation.velocity");
    optrot_frequency = input.getDblVec("OpticalRotation.frequency");
    optrot_perturbation = input.get<string>("OpticalRotation.perturbation");

    mol_charge = input.get<int>("Molecule.charge");
    mol_multiplicity = input.get<int>("Molecule.multiplicity");
    mol_coords = input.getData("Molecule.coords");

    wf_solvent = input.get<bool>("WaveFunction.solvent");
    wf_restricted = input.get<bool>("WaveFunction.restricted");
    wf_method = input.get<string>("WaveFunction.method");

    if (wf_method == "DFT") {
        dft_spin = input.get<bool>("DFT.spin");
        dft_x_fac = input.get<double>("DFT.exact_exchange");
        dft_cutoff = input.get<double>("DFT.density_cutoff");
        dft_func_coefs = input.getDblVec("DFT.func_coefs");
        dft_func_names = input.getData("DFT.functionals");
    }

    scf_run = input.get<bool>("SCF.run");
    scf_start = input.get<string>("SCF.initial_guess");
    scf_kain = input.get<int>("SCF.kain");
    scf_max_iter = input.get<int>("SCF.max_iter");
    scf_rotation = input.get<int>("SCF.rotation");
    scf_canonical = input.get<bool>("SCF.canonical");
    scf_write_orbitals = input.get<bool>("SCF.write_orbitals");
    scf_orbital_thrs = input.get<double>("SCF.orbital_thrs");
    scf_property_thrs = input.get<double>("SCF.property_thrs");
    scf_lambda_thrs = input.get<double>("SCF.lambda_thrs");
    scf_orbital_prec = input.getDblVec("SCF.orbital_prec");

    kin_free_run = input.get<bool>("KineticFree.run");
    kin_free_max_iter = input.get<int>("KineticFree.max_iter");
    kin_free_canonical = input.get<bool>("KineticFree.canonical");
    kin_free_orb_thrs = input.get<double>("KineticFree.orbital_thrs");
    kin_free_prop_thrs = input.get<double>("KineticFree.property_thrs");

    rsp_run = input.get<bool>("Response.run");
    rsp_start = input.get<string>("Response.initial_guess");
    rsp_kain = input.get<int>("Response.kain");
    rsp_max_iter = input.get<int>("Response.max_iter");
    rsp_canonical = input.get<bool>("Response.canonical");
    rsp_write_orbitals = input.get<bool>("Response.write_orbitals");
    rsp_orbital_thrs = input.get<double>("Response.orbital_thrs");
    rsp_property_thrs = input.get<double>("Response.property_thrs");
    rsp_directions = input.getIntVec("Response.directions");
    rsp_orbital_prec = input.getDblVec("Response.orbital_prec");

    file_start_orbitals = input.get<string>("Files.start_orbitals");
    file_final_orbitals = input.get<string>("Files.final_orbitals");
    file_basis_set = input.get<string>("Files.basis_set");
    file_dens_mat = input.get<string>("Files.dens_mat");
    file_fock_mat = input.get<string>("Files.fock_mat");
    file_energy_vec = input.get<string>("Files.energy_vec");
    file_mo_mat_a = input.get<string>("Files.mo_mat_a");
    file_mo_mat_b = input.get<string>("Files.mo_mat_b");

    r_O[0] = 0.0;
    r_O[1] = 0.0;
    r_O[2] = 0.0;

    helmholtz = 0;
    kain = 0;
    kain_x = 0;
    kain_y = 0;

    P = 0;
    PH_1 = 0;
    PH_2 = 0;
    ABGV_00 = 0;
    ABGV_55 = 0;

    molecule = 0;
    nuclei = 0;
    phi = 0;

    T = 0;
    V = 0;
    J = 0;
    K = 0;
    XC = 0;
    U_r = 0;
    fock = 0;

    phi_np1 = 0;
    J_np1 = 0;
    K_np1 = 0;
    XC_np1 = 0;
    fock_np1 = 0;

    phi_x = 0;
    phi_y = 0;
    dJ = 0;
    dK = 0;
    dXC = 0;
    d_fock = 0;

    xcfun = 0;

    h_E = 0;
    h_B = 0;
    h_M = 0;
}

bool SCFDriver::sanityCheck() const {
    if (wf_method == "DFT" and dft_spin) {
        MSG_ERROR("Spin DFT not implemented");
        return false;
    }
    if (wf_restricted and mol_multiplicity != 1) {
        MSG_ERROR("Restricted open-shell not implemented");
        return false;
    }
    if (calc_quadrupole_moment) {
        MSG_ERROR("Quadrupole moment not implemented");
        return false;
    }
    if (calc_polarizability) {
        MSG_ERROR("Polarizability not implemented");
        return false;
    }
    if (calc_hyperpolarizability) {
        MSG_ERROR("Hyperpolarizability not implemented");
        return false;
    }
    if (calc_optical_rotation) {
        MSG_ERROR("Optical rotation not implemented");
        return false;
    }
    if (calc_magnetizability) {
        MSG_ERROR("Magnetizability not implemented");
        return false;
    }
    if (calc_nmr_shielding) {
        MSG_ERROR("NMR shielding not implemented");
        return false;
    }
    if (calc_spin_spin_coupling) {
        MSG_ERROR("Spin-spin coupling not implemented");
        return false;
    }
    if (calc_hyperfine_coupling) {
        MSG_ERROR("Hyperfine coupling not implemented");
        return false;
    }
    return true;
}

void SCFDriver::setup() {
    // Setting up molecule
    molecule = new Molecule(mol_coords, mol_charge);
    int nEl = molecule->getNElectrons();
    nuclei = &molecule->getNuclei();

    // Setting up empty orbitals
    phi = new OrbitalVector(nEl, mol_multiplicity, wf_restricted);

    // Defining gauge origin
    const double *COM = molecule->getCenterOfMass();
    if (center_of_mass) {
        r_O[0] = COM[0];
        r_O[1] = COM[1];
        r_O[2] = COM[2];
    } else {
        r_O[0] = gauge[0];
        r_O[1] = gauge[1];
        r_O[2] = gauge[2];
    }

    // Setting up MW operators
    P = new PoissonOperator(*MRA, rel_prec);
    PH_1 = new PHOperator<3>(*MRA, 1); // first derivative
    PH_2 = new PHOperator<3>(*MRA, 2); // second derivative
    ABGV_00 = new ABGVOperator<3>(*MRA, 0.0, 0.0);
    ABGV_55 = new ABGVOperator<3>(*MRA, 0.5, 0.5);

    // Setting up perturbation operators
    int nNucs = molecule->getNNuclei();
    h_E = new H_E_dip(r_O);
    if (diff_orb == "PH")      h_B = new H_B_dip(*PH_1, r_O);
    if (diff_orb == "ABGV_00") h_B = new H_B_dip(*ABGV_00, r_O);
    if (diff_orb == "ABGV_55") h_B = new H_B_dip(*ABGV_55, r_O);
    h_M = new H_M_pso*[nNucs];
    for (int k = 0; k < nNucs; k++) {
        const double *r_K = molecule->getNucleus(k).getCoord();
        if (diff_pso == "PH")      h_M[k] = new H_M_pso(*PH_1, r_K);
        if (diff_pso == "ABGV_00") h_M[k] = new H_M_pso(*ABGV_00, r_K);
        if (diff_pso == "ABGV_55") h_M[k] = new H_M_pso(*ABGV_55, r_K);
    }

    // Setting up properties
    if (nmr_nucleus_k[0] < 0) {
        nmr_nucleus_k.clear();
        for (int k = 0; k < nNucs; k++) {
            nmr_nucleus_k.push_back(k);
        }
    }
    if (sscc_nucleus_k[0] < 0) {
        sscc_nucleus_k.clear();
        for (int k = 0; k < nNucs; k++) {
            sscc_nucleus_k.push_back(k);
        }
    }
    if (sscc_nucleus_l[0] < 0) {
        sscc_nucleus_l.clear();
        for (int l = 0; l < nNucs; l++) {
            sscc_nucleus_l.push_back(l);
        }
    }
    if (hfcc_nucleus_k[0] < 0) {
        hfcc_nucleus_k.clear();
        for (int k = 0; k < nNucs; k++) {
            hfcc_nucleus_k.push_back(k);
        }
    }

    if (calc_scf_energy) molecule->initSCFEnergy();
    if (calc_dipole_moment) molecule->initDipoleMoment();
    if (calc_quadrupole_moment) molecule->initQuadrupoleMoment();
    if (calc_magnetizability) {
        molecule->initMagnetizability();
        for (int d = 0; d < 3; d++) {
            if (rsp_directions[d] == 0) continue;
            rsp_calculations.push_back(h_B, 0.0, true, d);
        }
    }
    if (calc_nmr_shielding) {
        for (int k = 0; k < nmr_nucleus_k.size(); k++) {
            int K = nmr_nucleus_k[k];
            molecule->initNMRShielding(K);
            if (nmr_perturbation == "B") {
                for (int d = 0; d < 3; d++) {
                    if (rsp_directions[d] == 0) continue;
                    rsp_calculations.push_back(h_B, 0.0, true, d);
                }
            } else {
                const double *r_K = molecule->getNucleus(K).getCoord();
                for (int d = 0; d < 3; d++) {
                    if (rsp_directions[d] == 0) continue;
                    rsp_calculations.push_back(h_M[K], 0.0, true, d);
                }
            }
        }
    }
    if (calc_hyperfine_coupling) {
        for (int k = 0; k < hfcc_nucleus_k.size(); k++) {
            int K = hfcc_nucleus_k[k];
            molecule->initHyperFineCoupling(K);
        }
    }
    if (calc_spin_spin_coupling) {
        for (int k = 0; k < sscc_nucleus_k.size(); k++) {
            int K = sscc_nucleus_k[k];
            for (int l = 0; l < sscc_nucleus_l.size(); l++) {
                int L = sscc_nucleus_l[l];
                if (K != L) molecule->initSpinSpinCoupling(K, L);
            }
        }
    }
    if (calc_polarizability) {
        for (int i = 0; i < pol_frequency.size(); i++) {
            double omega = pol_frequency[i];
            molecule->initPolarizability(omega);
            NOT_IMPLEMENTED_ABORT;
        }
    }
    if (calc_optical_rotation) {
        for (int i = 0; i < optrot_frequency.size(); i++) {
            double omega = optrot_frequency[i];
            molecule->initOpticalRotation(omega);
            NOT_IMPLEMENTED_ABORT;
        }
    }

    // Setting up SCF
    helmholtz = new HelmholtzOperatorSet(rel_prec, scf_lambda_thrs);
    if (scf_kain > 0) kain = new KAIN(scf_kain);
    if (rsp_kain > 0) kain_x = new KAIN(rsp_kain);
    if (rsp_kain > 0) kain_y = new KAIN(rsp_kain);

    // Setting up Fock operator
    if (diff_kin == "PH")      T = new KineticOperator(*PH_1);
    if (diff_kin == "ABGV_00") T = new KineticOperator(*ABGV_00);
    if (diff_kin == "ABGV_55") T = new KineticOperator(*ABGV_55);
    V = new NuclearPotential(*nuclei, nuc_prec);

    //cavity/cavity inverse
    if (wf_solvent) {
        cavity = new CavityFunction(*nuclei, sol_slope, sol_eps_0, sol_eps_inf, sol_alpha);
        U_r = new ReactionPotential(sol_kain, rel_prec, *P, *ABGV_00, *cavity, *nuclei, *phi);
    }

    if (wf_method == "Core") {
        fock = new CoreHamiltonian(*T, *V, U_r);
    } else if (wf_method == "Hartree") {
        J = new CoulombPotential(*P, *phi);
        fock = new Hartree(*T, *V, *J, U_r);
    } else if (wf_method == "HF") {
        J = new CoulombPotential(*P, *phi);
        K = new ExchangePotential(*P, *phi);
        fock = new HartreeFock(*T, *V, *J, *K, U_r);
    } else if (wf_method == "DFT") {
        J = new CoulombPotential(*P, *phi);
        xcfun = new XCFunctional(dft_spin, dft_cutoff);
        for (int i = 0; i < dft_func_names.size(); i++) {
            xcfun->setFunctional(dft_func_names[i], dft_func_coefs[i]);
        }
        XC = new XCPotential(*xcfun, *phi, ABGV_00);
        if (dft_x_fac > MachineZero) {
            K = new ExchangePotential(*P, *phi, dft_x_fac);
        }
        fock = new DFT(*T, *V, *J, *XC, K, U_r);
    } else {
        MSG_ERROR("Invalid method");
    }
}

void SCFDriver::clear() {
    for (int k = 0; k < molecule->getNNuclei(); k++) {
        if (h_M[k] != 0) delete h_M[k];
    }
    if (h_M != 0) delete[] h_M;
    if (h_B != 0) delete h_B;
    if (h_E != 0) delete h_E;

    if (xcfun != 0) delete xcfun;

    if (fock != 0) delete fock;
    if (U_r != 0) delete U_r;
    if (XC != 0) delete XC;
    if (K != 0) delete K;
    if (J != 0) delete J;
    if (V != 0) delete V;
    if (T != 0) delete T;

    if (phi != 0) delete phi;
    if (molecule != 0) delete molecule;

    if (ABGV_55 != 0) delete ABGV_55;
    if (ABGV_00 != 0) delete ABGV_00;
    if (PH_2 != 0) delete PH_2;
    if (PH_1 != 0) delete PH_1;
    if (P != 0) delete P;

    if (kain != 0) delete kain;
    if (kain_x != 0) delete kain_x;
    if (kain_y != 0) delete kain_y;
    if (helmholtz != 0) delete helmholtz;
}

/** Setup n+1 Fock operator for energy optimization */
void SCFDriver::setup_np1() {
    phi_np1 = new OrbitalVector(*phi);

    if (wf_method == "Core") {
    } else if (wf_method == "Hartree") {
        J_np1 = new CoulombPotential(*P, *phi_np1);
    } else if (wf_method == "HF") {
        J_np1 = new CoulombPotential(*P, *phi_np1);
        K_np1 = new ExchangePotential(*P, *phi_np1);
    } else if (wf_method == "DFT") {
        J_np1 = new CoulombPotential(*P, *phi_np1);
        XC_np1 = new XCPotential(*xcfun, *phi_np1, ABGV_00);
        if (dft_x_fac > MachineZero) {
            K_np1 = new ExchangePotential(*P, *phi_np1, dft_x_fac);
        }
    } else {
        MSG_ERROR("Invalid method");
    }

    fock_np1 = new FockOperator(0, V, J_np1, K_np1, XC_np1);
}

void SCFDriver::clear_np1() {
    if (fock_np1 != 0) delete fock_np1;
    if (XC_np1 != 0) delete XC_np1;
    if (K_np1 != 0) delete K_np1;
    if (J_np1 != 0) delete J_np1;
    if (phi_np1 != 0) delete phi_np1;
}

void SCFDriver::setupInitialGroundState() {
    // Reading initial guess
    if (scf_start == "none") {
        // Project minimal basis set of hydrogen orbitals
        OrbitalProjector OP(rel_prec, max_scale);
        OrbitalVector *tmp = OP(*nuclei);

        // Compute orthonormalization matrix
        IdentityOperator I;
        I.setup(rel_prec);
        MatrixXd S = I(*tmp, *tmp);
        MatrixXd S_m12 = MathUtils::hermitianMatrixPow(S, -1.0/2.0);
        I.clear();

        // Compute core Hamiltonian matrix
        CoreHamiltonian h(*T, *V);
        h.setup(rel_prec);
        MatrixXd f_mat = h(*tmp, *tmp);
        h.clear();

        // Diagonalize core Hamiltonian matrix
        MatrixXd M = MathUtils::diagonalizeHermitianMatrix(f_mat);
        MatrixXd U = M.transpose()*S_m12;

        // Duplicate orbitals if unrestricted
        extendRotationMatrix(*phi, U);

        // Rotate n lowest energy orbitals of U*tmp into phi
        OrbitalAdder add(rel_prec, max_scale);
        add.rotate(*phi, U, *tmp);
        delete tmp;

    } else if (scf_start == "gto") {
        OrbitalProjector OP(rel_prec, max_scale);
        if (wf_restricted) {
            OP(*phi, file_basis_set, file_mo_mat_a);
        } else {
            OP(*phi, file_basis_set, file_mo_mat_a, file_mo_mat_b);
        }
    } else if (scf_start == "mw") {
        NOT_IMPLEMENTED_ABORT;
    } else {
        NOT_IMPLEMENTED_ABORT;
    }
}

OrbitalOptimizer* SCFDriver::setupOrbitalOptimizer() {
    if (helmholtz == 0) MSG_ERROR("Helmholtz operators not initialized");

    OrbitalOptimizer *optimizer = new OrbitalOptimizer(*helmholtz, kain);
    optimizer->setMaxIterations(scf_max_iter);
    optimizer->setRotation(scf_rotation);
    optimizer->setCanonical(scf_canonical);
    optimizer->setThreshold(scf_orbital_thrs, scf_property_thrs);
    optimizer->setOrbitalPrec(scf_orbital_prec[0], scf_orbital_prec[1]);

    return optimizer;
}

EnergyOptimizer* SCFDriver::setupEnergyOptimizer() {
    if (helmholtz == 0) MSG_ERROR("Helmholtz operators not initialized");

    EnergyOptimizer *optimizer = new EnergyOptimizer(*helmholtz);
    optimizer->setMaxIterations(kin_free_max_iter);
    optimizer->setCanonical(kin_free_canonical);
    optimizer->setThreshold(kin_free_orb_thrs, kin_free_prop_thrs);
    optimizer->setOrbitalPrec(rel_prec, rel_prec);

    return optimizer;
}

LinearResponseSolver* SCFDriver::setupLinearResponseSolver(bool dynamic) {
    if (helmholtz == 0) MSG_ERROR("Helmholtz operators not initialized");

    LinearResponseSolver *lrs = 0;
    if (dynamic) {
        lrs = new LinearResponseSolver(*helmholtz, kain_x, kain_y);
    } else {
        lrs = new LinearResponseSolver(*helmholtz, kain_x);
    }
    lrs->setMaxIterations(rsp_max_iter);
    lrs->setThreshold(rsp_orbital_thrs, rsp_property_thrs);
    lrs->setOrbitalPrec(rsp_orbital_prec[0], rsp_orbital_prec[1]);
    lrs->setupUnperturbed(rsp_orbital_prec[1], *fock, *phi, F);

    return lrs;
}

void SCFDriver::setupPerturbedOrbitals(bool dynamic) {
    if (phi == 0) MSG_ERROR("Orbitals not initialized");

    phi_x = new OrbitalVector(*phi);
    if (dynamic) {
        phi_y = new OrbitalVector(*phi);
    } else {
        phi_y = phi_x;
    }
}

void SCFDriver::clearPerturbedOrbitals(bool dynamic) {
    if (not dynamic) phi_y = 0;
    if (phi_x != 0) delete phi_x;
    if (phi_y != 0) delete phi_y;
    phi_x = 0;
    phi_y = 0;
}

void SCFDriver::setupPerturbedOperators(const ResponseCalculation &rsp_calc) {
    if (phi == 0) MSG_ERROR("Orbitals not initialized");
    if (phi_x == 0) MSG_ERROR("X orbitals not initialized");
    if (phi_y == 0) MSG_ERROR("Y orbitals not initialized");

    double xFac = 0.0;
    if (wf_method == "HF") {
        xFac = 1.0;
    } else if (wf_method == "DFT") {
        xFac = dft_x_fac;
    }
    if (xFac > MachineZero) {
        NOT_IMPLEMENTED_ABORT;
    }

    int d = rsp_calc.dir;
    RankOneTensorOperator<3> &dH = *rsp_calc.pert;
    if (not rsp_calc.isImaginary() or rsp_calc.isDynamic()) {
        NOT_IMPLEMENTED_ABORT;
    }

    d_fock = new FockOperator(0, 0, dJ, dK, dXC);
    d_fock->setPerturbationOperator(dH[d]);
}

void SCFDriver::clearPerturbedOperators() {
    if (d_fock != 0) delete d_fock;
    if (dXC != 0) delete dXC;
    if (dK != 0) delete dK;
    if (dJ != 0) delete dJ;

    d_fock = 0;
    dXC = 0;
    dK = 0;
    dJ = 0;
}


void SCFDriver::run() {
    if (not sanityCheck()) return;

    bool converged = runGroundState();
    if (converged) {
        for (int i = 0; i < rsp_calculations.size(); i++) {
            runLinearResponse(rsp_calculations[i]);
        }
    }

    printEigenvalues(*phi, F);
    molecule->printGeometry();
    molecule->printProperties();
    if (U_r != 0) {
        double spill_in = U_r->getSpillIn();
        double spill_out = U_r->getSpillOut();
        println(0, "                                                            ");
        println(0, "============================================================");
        println(0, "                         Spillover                          ");
        println(0, "------------------------------------------------------------");
        println(0, "                                                            ");
        println(0, " In                          " << std::setw(30) << spill_in  );
        println(0, " Out                         " << std::setw(30) << spill_out );
        println(0, "                                                            ");
        println(0, "============================================================");
        println(0, "                                                            ");
    }
}

bool SCFDriver::runGroundState() {
    if (phi == 0) MSG_ERROR("Orbitals not initialized");
    if (fock == 0) MSG_ERROR("Fock operator not initialized");
    bool converged = true;

    // Setup initial guess
    setupInitialGroundState();

    // Optimize orbitals
    if (scf_run) {
        OrbitalOptimizer *solver = setupOrbitalOptimizer();
        solver->setup(*fock, *phi, F);
        converged = solver->optimize();
        solver->clear();
        delete solver;
    } else {
        fock->setup(rel_prec);
        F = (*fock)(*phi, *phi);
        fock->clear();
    }

    // Optimize energy
    if (kin_free_run) {
        setup_np1();

        EnergyOptimizer *solver = setupEnergyOptimizer();
        solver->setup(*fock, *phi, F, *fock_np1, *phi_np1);
        converged = solver->optimize();
        solver->clear();

        clear_np1();
        delete solver;
    }

    if (scf_write_orbitals) NOT_IMPLEMENTED_ABORT;

    // Compute requested properties
    if (converged) calcGroundStateProperties();

    return converged;
}

void SCFDriver::runLinearResponse(const ResponseCalculation &rsp_calc) {
    double omega = rsp_calc.freq;
    bool dynamic = false;
    if (fabs(omega) > MachineZero) dynamic = true;
    setupPerturbedOrbitals(dynamic);
    setupPerturbedOperators(rsp_calc);

    bool converged = true;
    if (rsp_run) {
        LinearResponseSolver *solver = setupLinearResponseSolver(dynamic);
        solver->setup(*d_fock, *phi_x);
        converged = solver->optimize();
        solver->clear();
        delete solver;
    }

    if (rsp_write_orbitals) NOT_IMPLEMENTED_ABORT;

    // Compute requested properties
    if (converged) calcLinearResponseProperties(rsp_calc);

    clearPerturbedOperators();
    clearPerturbedOrbitals(dynamic);
}

void SCFDriver::calcGroundStateProperties() {
    if (calc_scf_energy) {
        fock->setup(rel_prec);
        TelePrompter::printHeader(0, "Calculating SCF energy");
        Timer timer;
        SCFEnergy &energy = molecule->getSCFEnergy();
        energy = fock->trace(*phi, F);
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
        fock->clear();
    }
    if (calc_dipole_moment) {
        TelePrompter::printHeader(0, "Calculating dipole moment");
        Timer timer;
        VectorXd &nuc = molecule->getDipoleMoment().getNuclear();
        VectorXd &el = molecule->getDipoleMoment().getElectronic();
        H_E_dip mu(r_O);
        mu.setup(rel_prec);
        nuc = mu.trace(*nuclei);
        el = mu.trace(*phi);
        mu.clear();
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
    }
    if (calc_magnetizability) {
        TelePrompter::printHeader(0, "Calculating diamagnetic magnetizability");
        Timer timer;
        MatrixXd &dia = molecule->getMagnetizability().getDiamagnetic();
        H_BB_dia h(r_O);
        h.setup(rel_prec);
        dia = -h.trace(*phi);
        h.clear();
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
    }
    if (calc_nmr_shielding) {
        TelePrompter::printHeader(0, "Calculating diamagnetic NMR shielding");
        Timer timer;
        for (int k = 0; k < nmr_nucleus_k.size(); k++) {
            int K = nmr_nucleus_k[k];
            NMRShielding &nmr = molecule->getNMRShielding(K);
            MatrixXd &dia = nmr.getDiamagnetic();
            const double *r_K = nmr.getNucleus().getCoord();
            H_BM_dia h(r_O, r_K);
            h.setup(rel_prec);
            dia = h.trace(*phi);
            h.clear();
        }
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
    }
    if (calc_spin_spin_coupling) {
        TelePrompter::printHeader(0, "Calculating diamagnetic spin-spin coupling");
        Timer timer;
        for (int k = 0; k < sscc_nucleus_k.size(); k++) {
            int K = sscc_nucleus_k[k];
            for (int l = 0; l < sscc_nucleus_l.size(); l++) {
                int L = sscc_nucleus_l[l];
                if (K == L) continue;
                SpinSpinCoupling &sscc = molecule->getSpinSpinCoupling(K, L);
                MatrixXd &dia = sscc.getDiamagnetic();
                const double *r_K = sscc.getNucleusK().getCoord();
                const double *r_L = sscc.getNucleusL().getCoord();
                NOT_IMPLEMENTED_ABORT;
            }
        }
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
    }
    if (calc_hyperfine_coupling) {
        TelePrompter::printHeader(0, "Calculating HyperFine Coupling Constant");
        Timer timer;

        for (int k = 0; k < hfcc_nucleus_k.size(); k++) {
            int K = hfcc_nucleus_k[k];
            HyperFineCoupling &hfc = molecule->getHyperFineCoupling(K);
            const Nuclei &nucs = molecule->getNuclei();
            const Nucleus &nuc = nucs[K];
            const double *r_K = nuc.getCoord();
            NOT_IMPLEMENTED_ABORT;
        }
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
    }
}

void SCFDriver::calcLinearResponseProperties(const ResponseCalculation &rsp_calc) {
    int j = rsp_calc.dir;

    if (calc_magnetizability and rsp_calc.pert == h_B) {
        TelePrompter::printHeader(0, "Calculating paramagnetic magnetizability");
        Timer timer;
        MatrixXd &para = molecule->getMagnetizability().getParamagnetic();
        h_B->setup(rel_prec);
        para.row(j) = -h_B->trace(*phi, *phi_x, *phi_y);
        h_B->clear();
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
    }
    if (calc_nmr_shielding) {
        if (nmr_perturbation == "B" and rsp_calc.pert == h_B) {
            Timer timer;
            TelePrompter::printHeader(0, "Calculating paramagnetic NMR shielding ");
            for (int k = 0; k < nmr_nucleus_k.size(); k++) {
                int K = nmr_nucleus_k[k];
                MatrixXd &para = molecule->getNMRShielding(K).getParamagnetic();
                h_M[K]->setup(rel_prec);
                para.row(j) = -h_M[K]->trace(*phi, *phi_x, *phi_y);
                h_M[K]->clear();
            }
            timer.stop();
            TelePrompter::printFooter(0, timer, 2);
        }
        if (nmr_perturbation == "M") {
            for (int k = 0; k < nmr_nucleus_k.size(); k++) {
                int K = nmr_nucleus_k[k];
                if (rsp_calc.pert == h_M[K]) {
                    Timer timer;
                    TelePrompter::printHeader(0, "Calculating paramagnetic NMR shielding");

                    MatrixXd &para = molecule->getNMRShielding(K).getParamagnetic();
                    h_B->setup(rel_prec);
                    para.col(j) = -h_B->trace(*phi, *phi_x, *phi_y);
                    h_B->clear();

                    timer.stop();
                    TelePrompter::printFooter(0, timer, 2);
                }
            }
        }
    }
    if (calc_spin_spin_coupling) {
        TelePrompter::printHeader(0, "Calculating paramagnetic spin-spin coupling");
        Timer timer;
        NOT_IMPLEMENTED_ABORT;
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
    }
}

void SCFDriver::printEigenvalues(OrbitalVector &orbs, MatrixXd &f_mat) {
    int oldprec = TelePrompter::setPrecision(5);
    TelePrompter::printHeader(0, "Fock matrix");
    println(0, f_mat);
    TelePrompter::printSeparator(0, '=', 2);

    TelePrompter::printHeader(0, "Orbital energies");
    println(0, "    n  spin  occ                            epsilon  ");
    TelePrompter::printSeparator(0, '-');
    SelfAdjointEigenSolver<MatrixXd> es(f_mat.cols());
    es.compute(f_mat);

    TelePrompter::setPrecision(15);
    VectorXd epsilon = es.eigenvalues();
    for (int i = 0; i < epsilon.size(); i++) {
        Orbital &orb = orbs.getOrbital(i);
        printout(0, setw(5) << i);
        printout(0, setw(5) << orb.printSpin());
        printout(0, setw(5) << orb.getOccupancy());
        printout(0, setw(44) << epsilon(i) << endl);
    }
    TelePrompter::printSeparator(0, '=', 2);
    TelePrompter::setPrecision(oldprec);
}

void SCFDriver::extendRotationMatrix(const OrbitalVector &orbs, MatrixXd &O) {
    int nPaired = orbs.getNPaired();
    int nAlpha  = orbs.getNAlpha();
    int nBeta   = orbs.getNBeta();
    int nCols   = O.cols();

    if (nBeta > nAlpha) {
        MSG_ERROR("Inconsistent orbital set: too many beta orbitals");
    }

    O.conservativeResize(nPaired + nAlpha + nBeta, NoChange);
    O.block(nPaired + nAlpha, 0, nBeta, nCols) = O.block(nPaired, 0, nBeta, nCols);
}
