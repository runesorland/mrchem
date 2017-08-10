/** \mainpage The MRCPP program
 *
 * \author Stig Rune Jensen
 *
 * \version 1.0
 *
 * \par Copyright:
 * GPLv4
 *
 */

#pragma GCC system_header
#include <Eigen/Core>

#include "mrchem.h"
#include "parallel.h"
#include "constants.h"
#include "MREnv.h"
#include "Timer.h"
#include "TelePrompter.h"
#include "MathUtils.h"
#include "Plot.h"
#include "Molecule.h"

#include "FunctionTree.h"
#include "ABGVOperator.h"
#include "PoissonOperator.h"
#include "HelmholtzOperator.h"
#include "MWProjector.h"
#include "MWAdder.h"
#include "MWMultiplier.h"
#include "MWDerivative.h"
#include "MWConvolution.h"
#include "GridGenerator.h"
//#include "GridCleaner.h"

#include "GaussFunc.h"

#include "CavityFunction.h"


using namespace std;
using namespace Eigen;

void testProjection();
void testAddition();
void testMultiplication();
void testDerivative();
void testPoisson();
void testSCF();
void testCavity();
void testSCFCavity();
//void testTreeCleaner();

Getkw Input;

template<int D> MultiResolutionAnalysis<D>* initializeMRA();
template<int D> GaussFunc<D>* initializeGauss(const double *pos = 0);

int main(int argc, char **argv) {
    Timer timer;
    MREnv::initializeMRCPP(argc, argv);

    bool run_projection = Input.get<bool>("Pilot.run_projection");
    bool run_addition = Input.get<bool>("Pilot.run_addition");
    bool run_multiplication = Input.get<bool>("Pilot.run_multiplication");
    bool run_derivative = Input.get<bool>("Pilot.run_derivative");
    bool run_poisson = Input.get<bool>("Pilot.run_poisson");
    bool run_scf = Input.get<bool>("Pilot.run_scf");
    bool run_cavity = Input.get<bool>("Pilot.run_cavity");
    bool run_scfcavity = Input.get<bool>("Pilot.run_scfcavity");
        
    
    if (run_projection) testProjection();
    if (run_addition) testAddition();
    if (run_multiplication) testMultiplication();
    if (run_derivative) testDerivative();
    if (run_poisson) testPoisson();
    if (run_scf) testSCF();
    if (run_cavity) testCavity();
    if (run_scfcavity) testSCFCavity();
    
    timer.stop();
    MREnv::finalizeMRCPP(timer);

    return 0;
}

template<int D>
MultiResolutionAnalysis<D>* initializeMRA() {
    // Constructing world box
    int min_scale = Input.get<int>("MRA.min_scale");
    vector<int> corner = Input.getIntVec("MRA.corner");
    vector<int> boxes = Input.getIntVec("MRA.boxes");
    NodeIndex<D> idx(min_scale, corner.data());
    BoundingBox<D> world(idx, boxes.data());

    // Constructing scaling basis
    int order = Input.get<int>("MRA.order");
    InterpolatingBasis basis(order);

    // Initializing MRA
    return new MultiResolutionAnalysis<D>(world, basis);
}

template<int D>
GaussFunc<D>* initializeGauss(const double *pos) {
    // Constructing analytic function
    double beta = 20.0;
    double alpha = pow(beta/pi, D/2.0);
    int pow[3] = {0, 0, 0};
    if (pos != 0) {
        return new GaussFunc<D>(beta, alpha, pos, pow);
    } else {
        double o[3] = {0.0, 0.0, 0.0};
        return new GaussFunc<D>(beta, alpha, o, pow);
    }
}

void testProjection() {
    Timer timer;
    TelePrompter::printHeader(0, "Testing MWProjector");

    int max_scale = MRA->getMaxScale();
    double prec = Input.get<double>("rel_prec");
    MWProjector<3> project(prec, max_scale);

    auto f = [] (const double *r) -> double {
        const double beta = 100.0;
        const double alpha = pow(beta/pi, 3.0/2.0);;
        const double r_0[3] = {0.0, 0.0, 0.0};
        double R = MathUtils::calcDistance(3, r, r_0);
        return alpha*exp(-beta*R*R);
    };

    FunctionTree<3> f_tree(*MRA);
    project(f_tree, f);

    double integral = f_tree.integrate();
    double sq_norm = f_tree.getSquareNorm();

    println(0, " Integral                    " << setw(30) << integral);
    println(0, " Square norm                 " << setw(30) << sq_norm);

    timer.stop();
    TelePrompter::printFooter(0, timer, 2);
}

void testAddition() {
    Timer timer;
    TelePrompter::printHeader(0, "Testing MWAdder");

    int max_scale = MRA->getMaxScale();
    double prec = Input.get<double>("rel_prec");
    MWProjector<3> project(prec, max_scale);
    MWAdder<3> add(prec, max_scale);

    double f_pos[3] = {0.0, 0.0,  0.1};
    double g_pos[3] = {0.0, 0.0, -0.1};
    GaussFunc<3> *f_func = initializeGauss<3>(f_pos);
    GaussFunc<3> *g_func = initializeGauss<3>(g_pos);

    FunctionTree<3> f_tree(*MRA);
    FunctionTree<3> g_tree(*MRA);
    FunctionTree<3> h_tree(*MRA);
    project(f_tree, *f_func);
    project(g_tree, *g_func);
    add(h_tree, 1.0, f_tree, -2.0, g_tree);

    double integral = h_tree.integrate();
    double sq_norm = h_tree.getSquareNorm();

    println(0, " Integral                    " << setw(30) << integral);
    println(0, " Square norm                 " << setw(30) << sq_norm);

    delete f_func;
    delete g_func;

    timer.stop();
    TelePrompter::printFooter(0, timer, 2);
}

void testMultiplication() {
    Timer timer;
    TelePrompter::printHeader(0, "Testing MWMultiplier");

    int max_scale = MRA->getMaxScale();
    double prec = Input.get<double>("rel_prec");
    MWProjector<3> project(prec, max_scale);
    MWMultiplier<3> mult(prec, max_scale);

    double f_pos[3] = {0.0, 0.0,  0.1};
    double g_pos[3] = {0.0, 0.0, -0.1};
    GaussFunc<3> *f_func = initializeGauss<3>(f_pos);
    GaussFunc<3> *g_func = initializeGauss<3>(g_pos);

    FunctionTree<3> f_tree(*MRA);
    FunctionTree<3> g_tree(*MRA);
    FunctionTree<3> h_tree(*MRA);
    project(f_tree, *f_func);
    project(g_tree, *g_func);
    mult(h_tree, 1.0, f_tree, g_tree);

    double integral = h_tree.integrate();
    double sq_norm = h_tree.getSquareNorm();

    println(0, " Integral                    " << setw(30) << integral);
    println(0, " Square norm                 " << setw(30) << sq_norm);

    delete f_func;
    delete g_func;

    timer.stop();
    TelePrompter::printFooter(0, timer, 2);
}

void testDerivative() {
    TelePrompter::printHeader(0, "Testing derivative operator");
    printout(0, endl);

    Timer tot_t;
    MultiResolutionAnalysis<1> *MRA_1 = initializeMRA<1>();

    int max_scale = MRA_1->getMaxScale();
    double prec = Input.get<double>("rel_prec");
    double proj_prec = prec/10.0;

    ABGVOperator<1> D(*MRA_1, 0.0, 0.0);
    MWDerivative<1> apply(max_scale);

    MWAdder<1> add(-1.0, max_scale);
    MWProjector<1> project(proj_prec, max_scale);

    auto f = [] (const double *r) -> double {
        const double alpha = 3.0;
        const double r_0[3] = {pi, pi, pi};
        double R = MathUtils::calcDistance(1, r, r_0);
        return exp(-alpha*R);
    };
    auto df = [] (const double *r) -> double {
        const double alpha = 3.0;
        const double r_0[3] = {pi, pi, pi};
        double R = MathUtils::calcDistance(1, r, r_0);
        double sign = 1.0;
        if (r[0] > r_0[0]) sign = -1.0;
        return sign*alpha*exp(-alpha*R);
    };

    Timer proj_t1;
    FunctionTree<1> f_tree(*MRA_1);
    project(f_tree, f);
    proj_t1.stop();
    println(0, " Projecting f      " << proj_t1);

    Timer proj_t2;
    FunctionTree<1> df_tree(*MRA_1);
    project(df_tree, df);
    proj_t2.stop();
    println(0, " Projecting df     " << proj_t2);

    Timer apply_t;
    FunctionTree<1> dg_tree(*MRA_1);
    apply(dg_tree, D, f_tree, 0); // Does not refine grid further
    apply_t.stop();
    println(0, " Applying D        " << apply_t);

    Timer add_t;
    FunctionTree<1> err_tree(*MRA_1);
    add(err_tree, 1.0, df_tree, -1.0, dg_tree);
    add_t.stop();
    println(0, " Computing error   " << add_t << endl);

    double f_int = f_tree.integrate();
    double f_norm = sqrt(f_tree.getSquareNorm());
    double df_int = df_tree.integrate();
    double df_norm = sqrt(df_tree.getSquareNorm());
    double dg_int = dg_tree.integrate();
    double dg_norm = sqrt(dg_tree.getSquareNorm());
    double abs_err = sqrt(err_tree.getSquareNorm());
    double rel_err = abs_err/df_norm;

    TelePrompter::printSeparator(0, '-', 1);
    println(0," f_tree integral:            " << setw(30) << f_int);
    println(0," f_tree norm:                " << setw(30) << f_norm << endl);
    println(0," df_tree integral:           " << setw(30) << df_int);
    println(0," df_tree norm:               " << setw(30) << df_norm << endl);
    println(0," dg_tree integral:           " << setw(30) << dg_int);
    println(0," dg_tree norm:               " << setw(30) << dg_norm << endl);
    println(0," absolute error:             " << setw(30) << abs_err);
    println(0," relative error:             " << setw(30) << rel_err << endl);

    tot_t.stop();
    TelePrompter::printFooter(0, tot_t, 2);
}

void testPoisson() {
    TelePrompter::printHeader(0, "Testing Poisson operator");
    printout(0, endl);

    Timer tot_t;
    int max_scale = MRA->getMaxScale();
    double prec = Input.get<double>("rel_prec");
    double proj_prec = prec/10.0;
    double build_prec = prec/10.0;
    double apply_prec = prec;

    GridGenerator<3> grid(max_scale);
    MWProjector<3> project(proj_prec, max_scale);

    double beta = 100.0;
    double alpha = pow(beta/pi, 3.0/2.0);
    double pos[3] = {pi/3.0,pi/3.0,pi/3.0};
    GaussFunc<3> f_func(beta, alpha, pos);

    TelePrompter::printHeader(0, "Computing analytic energy");
    Timer analy_t;
    double ana_energy = f_func.calcCoulombEnergy(f_func);
    analy_t.stop();
    TelePrompter::printFooter(0, analy_t, 2);

    TelePrompter::printHeader(0, "Projecting function");
    Timer proj_t;
    FunctionTree<3> f_tree(*MRA);
    grid(f_tree, f_func);
    project(f_tree, f_func);
    proj_t.stop();
    TelePrompter::printFooter(0, proj_t, 2);

    TelePrompter::printHeader(0, "Constructing Poisson operator");
    Timer build_t;
    PoissonOperator P(*MRA, build_prec);
    build_t.stop();
    TelePrompter::printFooter(0, build_t, 2);

    TelePrompter::printHeader(0, "Applying Poisson operator");
    Timer apply_t;
    MWConvolution<3> apply(apply_prec, MRA->getMaxScale());
    FunctionTree<3> g_tree(*MRA);
    apply(g_tree, P, f_tree);
    apply_t.stop();
    TelePrompter::printFooter(0, apply_t, 2);

    double f_int = f_tree.integrate();
    double f_norm = sqrt(f_tree.getSquareNorm());
    double g_int = g_tree.integrate();
    double g_norm = sqrt(g_tree.getSquareNorm());
    double num_energy = g_tree.dot(f_tree);
    double error = (num_energy-ana_energy)/num_energy;

    println(0, endl);
    println(0," f_tree integral:            " << setw(30) << f_int);
    println(0," f_tree norm:                " << setw(30) << f_norm << endl);
    println(0," g_tree integral:            " << setw(30) << g_int);
    println(0," g_tree norm:                " << setw(30) << g_norm << endl);
    println(0," Analytic energy:            " << setw(30) << ana_energy);
    println(0," Numerical energy:           " << setw(30) << num_energy);
    println(0," Relative error:             " << setw(30) << error << endl);
    println(0, endl);

    tot_t.stop();
    TelePrompter::printFooter(0, tot_t, 2);
}

void testSCF() {
    // Precision parameters
    int max_scale = MRA->getMaxScale();
    double prec = Input.get<double>("rel_prec");

    // Initializing projector
    GridGenerator<3> grid(max_scale);
    MWProjector<3> project(prec, max_scale);

    // Nuclear parameters
    double Z = 1.0;                     // Nuclear charge
    double R[3] = {0.0, 0.0, 0.0};      // Nuclear position

    // Orbtial energies
    double epsilon_n = -0.5;
    double epsilon_np1 = 0.0;
    double d_epsilon_n = 0.0;

    // Nuclear potential
    FunctionTree<3> V(*MRA);
    {
        Timer timer;
        int oldlevel = TelePrompter::setPrintLevel(10);
        TelePrompter::printHeader(0, "Projecting nuclear potential");

        double c = 0.00435*prec/pow(Z, 5);  // Smoothing parameter
        auto u = [] (double r) -> double {
            return erf(r)/r + 1.0/(3.0*sqrt(pi))*(exp(-r*r) + 16.0*exp(-4.0*r*r));
        };
        auto f = [u, c, Z, R] (const double *r) -> double {
            double x = MathUtils::calcDistance(3, r, R);
            return -1.0*Z*u(x/c)/c;
        };

        project(V, f);
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
        TelePrompter::setPrintLevel(oldlevel);
    }
    
    // Wave function
    FunctionTree<3> *phi_n = new FunctionTree<3>(*MRA);
    FunctionTree<3> *phi_np1 = 0;
    {
        Timer timer;
        int oldlevel = TelePrompter::setPrintLevel(10);
        TelePrompter::printHeader(0, "Projecting initial guess");

        auto f = [R] (const double *r) -> double {
            double x = MathUtils::calcDistance(3, r, R);
            return 1.0*exp(-1.0*x*x);
        };

        project(*phi_n, f);
        phi_n->normalize();
        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
        TelePrompter::setPrintLevel(oldlevel);
    }

    TelePrompter::printHeader(0, "Running SCF");
    printout(0, " Iter");
    printout(0, "      E_np1          dE_n   ");
    printout(0, "   ||phi_np1||   ||dPhi_n||" << endl);
    TelePrompter::printSeparator(0, '-');

    double scf_prec = 1.0e-3;
    double scf_thrs = prec*10.0;

    int iter = 1;
    double error = 1.0;
    vector<Timer> scf_t;
    while (error > scf_thrs) {
        Timer cycle_t;

        // Adjust precision
        scf_prec = min(scf_prec, error/100.0);
        scf_prec = max(scf_prec, prec);
        
        // Initialize Helmholtz operator
        if (epsilon_n > 0.0) epsilon_n *= -1.0;
        double mu_n = sqrt(-2.0*epsilon_n);
        HelmholtzOperator H(*MRA, mu_n, scf_prec);

        // Initialize arithmetic operators
        MWAdder<3> add(scf_prec, max_scale);
        MWMultiplier<3> mult(scf_prec, max_scale);
        MWConvolution<3> apply(scf_prec, max_scale);

        // Compute Helmholtz argument V*phi
        FunctionTree<3> Vphi(*MRA);
        grid(Vphi, *phi_n);  // Copy grid from orbital
        mult(Vphi, 1.0, V, *phi_n, 1);    // Relax grid max one level

        // Apply Helmholtz operator phi^n+1 = H[V*phi^n]
        phi_np1 = new FunctionTree<3>(*MRA);
        apply(*phi_np1, H, Vphi);
        *phi_np1 *= -1.0/(2.0*pi);

        // Compute orbital residual
        FunctionTree<3> d_phi_n(*MRA);
        grid(d_phi_n, *phi_np1);                      // Copy grid from phi_np1
        add(d_phi_n, 1.0, *phi_np1, -1.0, *phi_n); // No grid relaxation
        error = sqrt(d_phi_n.getSquareNorm());

        // Compute energy update
        d_epsilon_n = Vphi.dot(d_phi_n)/phi_np1->getSquareNorm();
        epsilon_np1 = epsilon_n + d_epsilon_n;

        printout(0, setw(3) << iter);
        TelePrompter::setPrecision(10);
        printout(0, setw(19) << epsilon_np1);
        TelePrompter::setPrecision(1);
        printout(0, setw(9) << d_epsilon_n);
        TelePrompter::setPrecision(10);
        printout(0, setw(19) << phi_np1->getSquareNorm());
        TelePrompter::setPrecision(1);
        printout(0, setw(9) << error);
        TelePrompter::setPrecision(15);
        printout(0, endl);

        delete phi_n;

        // Prepare for next iteration
        epsilon_n = epsilon_np1;
        phi_n = phi_np1;
        phi_n->normalize();
    
        double a[3] = { -4.0, -4.0, 0.0};
        double b[3] = { 4.0, 4.0, 0.0};
        Plot<3> plt(10000, a, b);
        plt.surfPlot(*phi_n, "phi_n");
     
        cycle_t.stop();
        scf_t.push_back(cycle_t);
        iter++;
    }
    TelePrompter::printSeparator(0, '=', 2);

    TelePrompter::printHeader(0, "SCF timings");
    for (int i = 0; i < scf_t.size(); i++) {
        println(0, " Time cycle " << setw(5) << i+1 << "  " << scf_t[i]);
    }
    TelePrompter::printSeparator(0, '=', 2);


    TelePrompter::setPrecision(15);
    TelePrompter::printHeader(0, "Final energy");
    println(0, " Orbital energy    " << setw(40) << epsilon_n);
    TelePrompter::printSeparator(0, '=', 2);

    delete phi_n;
}





//nuclear potential function
FunctionTree<3>* nuclearPotential(double prec, const Nuclei &nucs) {

    Timer timer;
    int oldlevel = TelePrompter::setPrintLevel(10);
    TelePrompter::printHeader(0, "Projecting nuclear potential");
   
    FunctionTree<3> *V_nuc = new FunctionTree<3>(*MRA);
    
            
    auto u = [] (double r) -> double {
        return erf(r)/r + 1.0/(3.0*sqrt(pi))*(exp(-r*r) + 16.0*exp(-4.0*r*r));
    };
   

    auto func = [u, prec, nucs] (const double* r) -> double{
        double func_val = 0.0;  
        for (int i = 0; i < nucs.size(); i++){
            double c = 0.0435*prec/pow(nucs[i].getCharge(),5);
            double x = MathUtils::calcDistance(3, r, nucs[i].getCoord());
            func_val += -1.0*nucs[i].getCharge()*u(x/c)/c;
        }
        return func_val;
    };  
   

    int max_scale = MRA->getMaxScale();
    MWProjector<3> project(prec, max_scale);
    project(*V_nuc, func);
    timer.stop();

    TelePrompter::printFooter(0, timer, 2);
    TelePrompter::setPrintLevel(oldlevel);

    return V_nuc;
}

//initial wavefunction estimate
// Wave function
FunctionTree<3>* initialWaveFunction(double prec, const Nuclei &nucs){
    
    Timer timer;
    int oldlevel = TelePrompter::setPrintLevel(10);
    TelePrompter::printHeader(0, "Projecting inital wavefunction");

    FunctionTree<3> *phi = new FunctionTree<3>(*MRA);

    auto func = [nucs] (const double *r) -> double{
        double func_val = 0.0;
        for (int i = 0; i < nucs.size(); i++){
            double x = MathUtils::calcDistance(3, r, nucs[i].getCoord());
            func_val += exp((-1.0*x*x)/nucs[i].getCharge());
        }
        return func_val;

    };
    
    int max_scale = MRA->getMaxScale();
    MWProjector<3> project(prec, max_scale);
    project(*phi, func);

    timer.stop();
    TelePrompter::printFooter(0, timer, 2);
    TelePrompter::setPrintLevel(oldlevel);

    return phi;
     

};






void testSCFCavity(){
    //Plotting parameters for surfPlot
    double a[3] = { -4.0, -4.0, 0.0};
    double b[3] = { 4.0, 4.0, 0.0};
    Plot<3> plt(10000, a, b);
        
    // Precision parameters
    int max_scale = MRA->getMaxScale();
    double prec = Input.get<double>("rel_prec");

    // Initializing projector
    GridGenerator<3> grid(max_scale);
    MWProjector<3> project(prec, max_scale);

    // Nuclear parameters
   // double Z = 1.0;                     // Nuclear charge
    //double R[3] = {0.0, 0.0, 0.0};      // Nuclear position
  
    //Importing molecular information.
    std::vector<std::string> mol_coords = Input.getData("Molecule.coords");
    Molecule mol(mol_coords);
    mol.printGeometry();

    // Orbtial energies
    double energy_n = -0.5;
    double energy_np1 = 0.0;
    double d_energy_n = 0.0;

   
    Nuclei &nucs = mol.getNuclei();
   
   
    // Wave function
    FunctionTree<3> *phi_n = initialWaveFunction(prec, nucs);
    FunctionTree<3> *phi_np1 = 0;
   // {
   //     Timer timer;
   //     int oldlevel = TelePrompter::setPrintLevel(10);
   //     TelePrompter::printHeader(0, "Projecting initial guess, phi");

   //     auto f = [R] (const double *r) -> double {
   //         double x = MathUtils::calcDistance(3, r, R);
   //         return 1.0*exp(-1.0*x*x);
   //     };

   //     project(*phi_n, f);
   //     phi_n->normalize();
   //     timer.stop();
   //     TelePrompter::printFooter(0, timer, 2);
   //     TelePrompter::setPrintLevel(oldlevel);
   // }
    
    
    
    //nuclear potential
    FunctionTree<3> *V_nuc = nuclearPotential(prec, nucs);
       
    //cavity
    FunctionTree<3> eps(*MRA);
    FunctionTree<3> eps_inv(*MRA);
    {
        Timer timer;
        int oldlevel = TelePrompter::setPrintLevel(10);
        TelePrompter::printHeader(0, "Projecting cavityfunction and cavityfunction inverse");

        double slope = 0.2;//slope of cavity, lower double --> steeper slope.
        double eps_0 = 1.0;
        double eps_inf = 10.0;

        CavityFunction cavity(nucs, slope, false, eps_0, eps_inf);
        project(eps, cavity);

        //cavity invers
        CavityFunction cavity_inv(nucs, slope, true, eps_0, eps_inf);
        project(eps_inv, cavity_inv);

        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
        TelePrompter::setPrintLevel(oldlevel);
    
    }


       
    //initializing operator

    double scf_prec = prec;
   
    //derivative
    double boundary1 = 0.0, boundary2 = 0.0;
    ABGVOperator<3> D(*MRA, boundary1, boundary2);
    MWDerivative<3> applyDerivative(max_scale); 
    
    // Poissoni/Greens
    PoissonOperator P(*MRA, scf_prec);

    // arithmetics 
    MWAdder<3> add(scf_prec, max_scale);
    MWMultiplier<3> mult(scf_prec, max_scale);
    MWConvolution<3> apply(scf_prec, max_scale);

    //derivative of cavity 
    FunctionTree<3> dx_eps(*MRA);
    FunctionTree<3> dy_eps(*MRA);
    FunctionTree<3> dz_eps(*MRA);
    
    applyDerivative(dx_eps, D, eps, 0);
    applyDerivative(dy_eps, D, eps, 1);
    applyDerivative(dz_eps, D, eps, 2);


    FunctionTree<3> *V = new FunctionTree<3>(*MRA);
    FunctionTree<3> *V_el_n = new FunctionTree<3>(*MRA);
    V_el_n->setZero();
    FunctionTree<3> *V_el_np1 = 0;
    
    
    
    TelePrompter::printHeader(0, "Running SCF");
    printout(0, " Iter");
    printout(0, "      E_np1          dE_n   ");
    printout(0, "   ||phi_np1||   ||dPhi_n||" << endl);
    TelePrompter::printSeparator(0, '-');
    


    int cycle = 0; 
    int iter = 1;
    double errorPhi = 1.0;
    double errorV = 1.0;
    vector<Timer> scf_t;
    while (errorPhi > scf_prec) {
        Timer cycle_t;    
        if (errorPhi < scf_prec*10){
            while (errorV > scf_prec){
            
                //derivative of electrostatic potential
                FunctionTree<3> dx_V_el(*MRA);
                FunctionTree<3> dy_V_el(*MRA);
                FunctionTree<3> dz_V_el(*MRA);
                
                  
                applyDerivative(dx_V_el, D, *V_el_n, 0);
                applyDerivative(dy_V_el, D, *V_el_n, 1);
                applyDerivative(dz_V_el, D, *V_el_n, 2);

                plt.surfPlot(dx_V_el, "dx_V_el"); 
                plt.surfPlot(dy_V_el, "dy_V_el"); 
                plt.surfPlot(dz_V_el, "dz_V_el"); 
                
                //creating rho_eff (rho/eps)
                FunctionTree<3> rho(*MRA);
                mult(rho, 1.0, *phi_n, *phi_n);
                
                FunctionTree<3> rho_eff(*MRA);
                mult(rho_eff, 1.0, rho, eps_inv);
               

                //creating gamma (grad_eps*grad_V)/4pi*eps)
                FunctionTreeVector<3> gradeps_gradV;
                FunctionTree<3> dx_eps_dx_V_el(*MRA);
                FunctionTree<3> dy_eps_dy_V_el(*MRA);
                FunctionTree<3> dz_eps_dz_V_el(*MRA);
                
                mult(dx_eps_dx_V_el, 1.0, dx_eps, dx_V_el);
                mult(dy_eps_dy_V_el, 1.0, dy_eps, dy_V_el);
                mult(dz_eps_dz_V_el, 1.0, dz_eps, dz_V_el);
                
                gradeps_gradV.push_back(1.0, &dx_eps_dx_V_el);
                gradeps_gradV.push_back(1.0, &dy_eps_dy_V_el);
                gradeps_gradV.push_back(1.0, &dz_eps_dz_V_el);

                FunctionTree<3> temp_func(*MRA);
                add(temp_func, gradeps_gradV);
                temp_func *= 1.0/(4.0*pi);  
                gradeps_gradV.clear();
                
                FunctionTree<3> gamma(*MRA);
                mult(gamma, 1.0, temp_func, eps_inv);

                
                FunctionTree<3> sum_rhoeff_gamma(*MRA);
                add(sum_rhoeff_gamma, 1.0, rho_eff, 1.0, gamma);


                plt.surfPlot(sum_rhoeff_gamma, "sum_rhoeff_gamma");
                       
                //applying greensfunction
                V_el_np1 = new FunctionTree<3>(*MRA);
                apply(*V_el_np1, P, sum_rhoeff_gamma);
            


                //preparing for next iteration
                FunctionTree<3> error_func(*MRA);
                add(error_func, 1.0, *V_el_n, -1.0, *V_el_np1);

                errorV = sqrt(error_func.getSquareNorm());
                
                delete V_el_n;
                V_el_n = V_el_np1; 
                cycle += 1;
                std::cout << cycle << ":  " << errorV << std::endl; 
                //Plotting
                plt.surfPlot(*V_el_n, "V_el_n");
                plt.surfPlot(gamma, "gamma");

            }
        }
        
        add(*V, 1.0, *V_el_n, 1.0, *V_nuc);


        // Helmholtz operator
        if (energy_n > 0.0) energy_n *= -1.0;
        double mu_n = sqrt(-2.0*energy_n);
        HelmholtzOperator H(*MRA, mu_n, scf_prec);


        // Compute Helmholtz argument V*phi
        FunctionTree<3> Vphi(*MRA);
        grid(Vphi, *phi_n);  // Copy grid from orbital
        mult(Vphi, 1.0, *V, *phi_n, 1);   
        
        
        // Apply Helmholtz operator phi^n+1 = H[V*phi^n]
        phi_np1 = new FunctionTree<3>(*MRA);
        apply(*phi_np1, H, Vphi);
        *phi_np1 *= -1.0/(2.0*pi);

        // Compute orbital residual
        FunctionTree<3> d_phi_n(*MRA);
        grid(d_phi_n, *phi_np1);                      // Copy grid from phi_np1
        add(d_phi_n, 1.0, *phi_np1, -1.0, *phi_n); // No grid relaxation
        
        errorPhi = sqrt(d_phi_n.getSquareNorm());
        errorV = 1;
        
        // Compute energy update
        d_energy_n = Vphi.dot(d_phi_n)/phi_np1->getSquareNorm();
        energy_np1 = energy_n + d_energy_n;

        printout(0, setw(3) << iter);
        TelePrompter::setPrecision(10);
        printout(0, setw(19) << energy_np1);
        TelePrompter::setPrecision(1);
        printout(0, setw(9) << d_energy_n);
        TelePrompter::setPrecision(10);
        printout(0, setw(19) << phi_np1->getSquareNorm());
        TelePrompter::setPrecision(1);
        printout(0, setw(9) << errorPhi);
        TelePrompter::setPrecision(15);
        printout(0, endl);


        

        // Prepare for next iteration
        energy_n = energy_np1;
        phi_n = phi_np1;
        phi_n->normalize();
        

               
        cycle_t.stop();
        scf_t.push_back(cycle_t);
        iter++;
        
    }
    

    
    TelePrompter::printSeparator(0, '=', 2);

    TelePrompter::printHeader(0, "SCF timings");
    for (int i = 0; i < scf_t.size(); i++) {
        println(0, " Time cycle " << setw(5) << i+1 << "  " << scf_t[i]);
    }
    TelePrompter::printSeparator(0, '=', 2);


    TelePrompter::setPrecision(15);
    TelePrompter::printHeader(0, "Final energy");
    println(0, " Orbital energy    " << setw(40) << energy_n);
    TelePrompter::printSeparator(0, '=', 2);

    delete phi_n;



}



void testCavity() {
/* 
Testing the Cavity class,(cavityFunction.h) and calculating the potential U

*author Rune
*/
    
    const int max_scale = MRA->getMaxScale();
    const double prec = Input.get<double>("rel_prec");
    const double build_prec = prec/10.0;
    const double apply_prec = prec;

    //Intitializing operators.
    GridGenerator<3> grid(max_scale);

    MWProjector<3> project(prec, max_scale);
    
    MWAdder<3> add(prec, max_scale); 
    
    MWMultiplier<3> mult(prec, max_scale);
    
    MWConvolution<3> apply_conv(apply_prec, MRA->getMaxScale());
    PoissonOperator P(*MRA, build_prec);
    
    double boundary1 = 0.0, boundary2 = 0.0;
    ABGVOperator<3> D(*MRA, boundary1, boundary2);
    MWDerivative<3> applyDerivative(max_scale);


    //Importing molecular information.
    std::vector<std::string> mol_coords = Input.getData("Molecule.coords");
    Molecule mol(mol_coords);
    mol.printGeometry();

    //Plotting parameters for surfPlot
    double a[3] = { -4.0, -4.0, 0.0};
    double b[3] = { 4.0, 4.0, 0.0};
    Plot<3> plt(10000, a, b);

    //nice to have, for testing. r and 1/r.
    auto f1  = [] (const double *r) ->double {
        double R = 1/sqrt(r[0]*r[0]+ r[1]*r[1] + r[2]*r[2]);
        return R;
    };
    
    FunctionTree<3> *one_over_rad = new FunctionTree<3>(*MRA);
    project(*one_over_rad, f1);
    
    plt.surfPlot(*one_over_rad, "one_over_rad");
    
    auto f2 = [] (const double *r) ->double {
        double R = sqrt(r[0]*r[0]+ r[1]*r[1] + r[2]*r[2]);
        return R;
    };
    
    FunctionTree<3> *rad = new FunctionTree<3>(*MRA);
    project(*rad,f2);
    
    plt.surfPlot(*rad, "rad");
    
    //making cavity
    Nuclei &nucs = mol.getNuclei();
    double slope = 0.2;//slope of cavity, lower double --> steeper slope.
    double eps_0 = 1.0;
    double eps_inf = 10.0;

    CavityFunction cavity(nucs,slope,false, eps_0, eps_inf);
    FunctionTree<3> *eps_r = new FunctionTree<3>(*MRA);//
    project(*eps_r, cavity);//

	//cavity invers
    CavityFunction cavity_inv(nucs,slope,true, eps_0, eps_inf);
    FunctionTree<3> *eps_r_inv = new FunctionTree<3>(*MRA);//
    project(*eps_r_inv, cavity_inv);//

    //making electron  density function
/*
should come from a HF calculation of.
Now it is a single Gaussian located with peeks
at (0,0,0) and (1,0,0), center of H atom from input file
*/ 
	//TODO should be from DFT calculations!		
    double alpha = 50.0;
    double c = pow(alpha/pi,3.0/2.0);
    double pos1[3] = {0, 0, 0};
    double pos2[3] = {1, 0, 0};
    int pov[3] = {0, 0, 0};
    GaussFunc<3> rho_r_analytic1(alpha, c , pos1, pov);
    GaussFunc<3> rho_r_analytic2(alpha, c , pos2, pov);
    
    FunctionTree<3> *rho_r1 = new FunctionTree<3>(*MRA);
    FunctionTree<3> *rho_r2 = new FunctionTree<3>(*MRA);
    project(*rho_r1, rho_r_analytic1);
    project(*rho_r2, rho_r_analytic2);
    FunctionTree<3> *rho_r = new FunctionTree<3>(*MRA);
    add(*rho_r, 1.0, *rho_r1, 1.0, *rho_r2);

    //derivation of dielectric function eps_r	
    FunctionTree<3> *dx_eps_r = new FunctionTree<3>(*MRA);//
    FunctionTree<3> *dy_eps_r = new FunctionTree<3>(*MRA);//
    FunctionTree<3> *dz_eps_r = new FunctionTree<3>(*MRA);//
   
    applyDerivative(*dx_eps_r, D, *eps_r,0);//
    applyDerivative(*dy_eps_r, D, *eps_r,1);//
    applyDerivative(*dz_eps_r, D, *eps_r,2);//
    
    
    //derivative vector of eps_r
    FunctionTreeVector<3> eps_r_derivative;
    eps_r_derivative.push_back(dx_eps_r);
    eps_r_derivative.push_back(dy_eps_r);
    eps_r_derivative.push_back(dz_eps_r);

    //making initial guess -rho/epsilon

    FunctionTree<3> *rho_eff = new FunctionTree<3>(*MRA);
    FunctionTreeVector<3> mult_initial_vec;
    //mult_initial_vec.push_back(1, &*eps_r_inv);
    //mult_initial_vec.push_back(1, &*rho_r);
    mult(*rho_eff, 1.0, *eps_r_inv, *rho_r);
    //mult(*rho_eff, mult_initial_vec);
    //FunctionTree<3> *rho_eff= mult(1, *eps_r_inv, *rho_r);

    plt.surfPlot(*rho_eff, "initial");
    TelePrompter::printHeader(0, "Constructing Poisson operator");
    

    TelePrompter::printHeader(0, "Applying Poisson operator");
    FunctionTree<3> *U = new FunctionTree<3>(*MRA);//
    apply_conv(*U, P, *rho_eff);

//SCF to solve for U, the potential.
    int iter = 1;
    double error = 1.0;
    vector<Timer> scf_t;
    double scf_thrs = prec;//prec, user defined
    while (error > scf_thrs) {
        println(0, "SCF: "<< iter);

        //making empty derivative functions.
        FunctionTree<3> *dx_U = new FunctionTree<3>(*MRA);
        FunctionTree<3> *dy_U = new FunctionTree<3>(*MRA);
        FunctionTree<3> *dz_U = new FunctionTree<3>(*MRA);

        grid(*dx_U,*U);
        grid(*dy_U,*U);
        grid(*dz_U,*U);       


        //doing the derivation
        applyDerivative(*dx_U, D, *U, 0);
        applyDerivative(*dy_U, D, *U, 1);
        applyDerivative(*dz_U, D, *U, 2);

        //vector to fill with u derivatives
        FunctionTreeVector<3> U_derivative;

        //filling the vector
        U_derivative.push_back(1, dx_U);
        U_derivative.push_back(1, dy_U);
        U_derivative.push_back(1, dz_U);

        //empty vector, filled with grad eps * grad U
        FunctionTreeVector<3> gradu_gradeps;
        /*
        for (int i = 0; i < 3; i++) {
           gradu_gradeps.push_back(mult(1,*U_derivative[i],*eps_r_derivative[i]));
        }
        */


        FunctionTreeVector<3> gradu_gradeps_x;
        gradu_gradeps_x.push_back(1, U_derivative[0]);
        gradu_gradeps_x.push_back(1, eps_r_derivative[0]);
        FunctionTree<3> *gradu_gradeps_0 = new FunctionTree<3>(*MRA);
        mult(*gradu_gradeps_0,gradu_gradeps_x);

        FunctionTreeVector<3> gradu_gradeps_y;
        gradu_gradeps_y.push_back(1, U_derivative[1]);
        gradu_gradeps_y.push_back(1, eps_r_derivative[1]);
        FunctionTree<3> *gradu_gradeps_1 = new FunctionTree<3>(*MRA); 
        mult(*gradu_gradeps_1,gradu_gradeps_y);

        FunctionTreeVector<3> gradu_gradeps_z;
        gradu_gradeps_z.push_back(1, U_derivative[2]);
        gradu_gradeps_z.push_back(1, eps_r_derivative[2]);
        FunctionTree<3> *gradu_gradeps_2 = new FunctionTree<3>(*MRA); 
        mult(*gradu_gradeps_2,gradu_gradeps_z);
        
       
        gradu_gradeps.push_back(1, gradu_gradeps_0);
        gradu_gradeps.push_back(1, gradu_gradeps_1);
        gradu_gradeps.push_back(1 ,gradu_gradeps_2);

        //sum up gradu_gradeps
        FunctionTree<3> *temp_func = new FunctionTree<3>(*MRA);
        add(*temp_func, gradu_gradeps);
        *temp_func *= 1.0/(4.0*pi);//OBS differnt form article!!
        
        FunctionTree<3> *gamma = new FunctionTree<3>(*MRA); 
        FunctionTreeVector<3> mult_gamma_vec;
        mult_gamma_vec.push_back(1, eps_r_inv);
        mult_gamma_vec.push_back(1, temp_func);

        mult(*gamma, mult_gamma_vec);

        plt.surfPlot(*gamma, "gamma");

        FunctionTree<3> *func = new FunctionTree<3>(*MRA); 
        FunctionTreeVector<3> sum_rho_gamma_vec;
        sum_rho_gamma_vec.push_back(1, rho_eff);
        sum_rho_gamma_vec.push_back(1, gamma);

        add(*func,sum_rho_gamma_vec);

        FunctionTree<3> *U_new = new FunctionTree<3>(*MRA);
        apply_conv(*U_new, P, *func);
                       

        //error calcultation, norm
        FunctionTree<3> *err = new FunctionTree<3>(*MRA); 
        FunctionTreeVector<3> difference_U_Unew;
        difference_U_Unew.push_back(1,U_new);
        difference_U_Unew.push_back(-1,U);

        add(*err,1.0,*U_new, -1.0, *U);
        error = sqrt(err->getSquareNorm());

        delete U;
        gradu_gradeps.clear(true);//fjerner ueps, koffor ikkje delete ueps.
        U = U_new;
        println(0,"error: " << error);
        delete dx_U;
        delete dy_U;
        delete dz_U;
        //delete U_derivative;error
        iter++;
/*
###############END#############
*/

////energy calculations

        FunctionTree<3> *temp = new FunctionTree<3>(*MRA);
        add(*temp, 1.0, *rho_eff, -1.0, *rho_r);

        FunctionTree<3> *gamma_rho_rho = new FunctionTree<3>(*MRA);
        add(*gamma_rho_rho, 1.0, *temp, 1.0, *gamma);

        FunctionTree<3> *U_r = new FunctionTree<3>(*MRA);
        apply_conv(*U_r, P , *gamma_rho_rho);

        FunctionTree<3> *U_v = new FunctionTree<3>(*MRA);
        apply_conv(*U_v, P, *rho_r);

        double E_r1 = U_r->dot(*rho_r); 
        double E_r2 = U_v->dot(*gamma_rho_rho); 
    
        println(0,"E_r1" << E_r1);
        println(0,"E_r2" << E_r2);   

//testing integrate rho_eff + gamma - rho, gamma, rho_eff-rho
/*
        FunctionTree<3> *pre_int_func = add(1.0, *rho_eff, -1.0, *rho_r);  
        FunctionTree<3> *int_func = add(1.0, *pre_int_func, 1.0, *gamma);
        plt.surfPlot(*int_func,"int_func");	
        double int_gamma = gamma->integrate();
        double charge = int_func->integrate();
        double int_rho_min_rho_eff = pre_int_func->integrate(); 
        double int_rho = rho_r->integrate();	
        println(0,"charge" << charge);
        println(0,"int_gamma" << int_gamma);
        println(0,"int_rho_min_rho_eff" << int_rho_min_rho_eff);		
        println(0,"int_rho" << int_rho);
*/

        
              
    }


/////////////Plotting and testing/////////
    ///energy

/* 
//For use with more than one nuclei, or one nuclei
//centered at not orego
    double c_list[nucs.size()];
    
    for (int i = 0; i < nucs.size(); i++) {
       const Nucleus &nuc = nucs[i];
       const double *coord = nuc.getCoord();
       double rad = nuc.getElement().getVdw();
    }
    double s = sqrt(pow(coord[0]-r[0],2)+pow(coord[1]-r[1],2)+pow(coord[2]-r[2],2)) - rad;

    plt.surfPlot(*eps_r,"eps_r");
    plt.surfPlot(*rho_r, "rho_r");
    plt.surfPlot(*eps_r_inv, "eps_r_inv");
    plt.surfPlot(*U,"U");
   
*/  
    double vdw = nucs[0].getElement().getVdw();    
    double E_ref = - (cavity.epsilon_inf-1.0)/(cavity.epsilon_inf * vdw);
    println(0,"E_ref" << E_ref);

    
    
    FunctionTree<3> *one_div_eps_mult_r = new FunctionTree<3>(*MRA);
    mult(*one_div_eps_mult_r, 1.0,*one_over_rad, *eps_r_inv);	
    plt.surfPlot(*one_div_eps_mult_r, "one_div_eps_mult_r");



	
    FunctionTreeVector<3> inp_vec2;
    inp_vec2.push_back(1.0, U);
    inp_vec2.push_back(1.0, eps_r);
    inp_vec2.push_back(1.0, rad);


    FunctionTree<3> *u_r_eps = new FunctionTree<3>(*MRA);
    mult(*u_r_eps, inp_vec2);
    plt.surfPlot(*u_r_eps,"u_r_eps");
    plt.surfPlot(*rad,"rad");
    plt.surfPlot(*eps_r,"eps_r");
	



    
    //delete prec; error
    //delete mol_coords; error
    delete one_over_rad;
    delete rad;
    //delete nucs;
    //delete slope;
    delete eps_r;
    //delete alpha;error
    //delete c;error
    //delete pos;error
    //delete pow;error
    //delete rho_r_analytic;error
    delete rho_r;
    //delete D_x; error
    //delete D_y; error
    //delete D_z; error
    delete dx_eps_r;
    delete dy_eps_r;
    delete dz_eps_r;
    //delete eps_r_derivative;error
    delete rho_eff;
    //delete P;
    delete U;
    
}    




/*
void testTreeCleaner() {
    Timer timer;
    TelePrompter::printHeader(0, "Testing TreeCleaner");

    GaussFunc<3> *f_func = initializeGauss<3>();

    double prec = Input.get<double>("rel_prec");
    GridCleaner<3> clean(prec, MRA->getMaxScale());
    MWProjector<3> project(-1.0, MRA->getMaxScale());

    FunctionTree<3> f_tree(*MRA);

    int n_nodes = 1;
    while (n_nodes > 0) {
        project(f_tree, *f_func);
        n_nodes = clean(f_tree);
    }
    project(f_tree, *f_func);

    double integral = f_tree.integrate();
    double sq_norm = f_tree.getSquareNorm();

    println(0, " Integral                    " << setw(30) << integral);
    println(0, " Square norm                 " << setw(30) << sq_norm);

    delete f_func;
    timer.stop();
    TelePrompter::printFooter(0, timer, 2);
}
*/

