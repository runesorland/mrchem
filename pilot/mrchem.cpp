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

//void testProjection();
//void testAddition();
//void testMultiplication();
//void testDerivative();
//void testPoisson();
//void testSCF();
void testCavity();
void testSCFCavity();
//void testTreeCleaner();

Getkw Input;

template<int D> MultiResolutionAnalysis<D>* initializeMRA();
template<int D> GaussFunc<D>* initializeGauss(const double *pos = 0);

int main(int argc, char **argv) {
    Timer timer;
    MREnv::initializeMRCPP(argc, argv);

    bool run_scfcavity = Input.get<bool>("Pilot.run_scfcavity");
    
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

FunctionTree<3>* nuclearChargeDensity(double prec, const Nuclei &nucs){
    
    Timer timer;
    int oldlevel = TelePrompter::setPrintLevel(10);
    TelePrompter::printHeader(0, "Projecting nuclear density");
   
    FunctionTree<3> *rho_nuc = new FunctionTree<3>(*MRA);
    
    double beta = pow(10.0, 6.0);
    double alpha = -nucs[0].getCharge()*pow((beta/pi),(3.0/2.0));
    GaussFunc<3> func(beta, alpha, nucs[0].getCoord());

    int max_scale = MRA->getMaxScale();
    MWProjector<3> project(prec, max_scale);
    GridGenerator<3> grid(max_scale);
    grid(*rho_nuc, func);
    project(*rho_nuc, func);
    timer.stop();

    TelePrompter::printFooter(0, timer, 2);
    TelePrompter::setPrintLevel(oldlevel);

    return rho_nuc;


    }   

//initial wavefunction estimate
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

double surfaceChargeTesting(
                FunctionTree<3> &epsilon, 
                FunctionTree<3> &epsilon_inv, 
                FunctionTree<3> &dx_epsilon, 
                FunctionTree<3> &dy_epsilon,
                FunctionTree<3> &dz_epsilon,
                FunctionTree<3> *V_g,
                FunctionTree<3> *V){
        
    //arithmetics 
    int max_scale = MRA->getMaxScale();
    double prec = Input.get<double>("rel_prec");
    MWAdder<3> add(prec, max_scale);
    MWMultiplier<3> mult(prec, max_scale);
        
    //creating gamma (grad_eps*grad_V)/(4pi*eps)
    FunctionTreeVector<3> gradeps_gradV;
    FunctionTree<3> dx_V_g(*MRA);
    FunctionTree<3> dy_V_g(*MRA);
    FunctionTree<3> dz_V_g(*MRA);
    FunctionTree<3> dx_eps_dx_V(*MRA);
    FunctionTree<3> dy_eps_dy_V(*MRA);
    FunctionTree<3> dz_eps_dz_V(*MRA);
   
    double boundary1 = 0.0, boundary2 = 0.0;
    ABGVOperator<3> D(*MRA, boundary1, boundary2);
    MWDerivative<3> applyDerivative(max_scale); 

    applyDerivative(dx_V_g, D, *V_g, 0);
    applyDerivative(dy_V_g, D, *V_g, 1);
    applyDerivative(dz_V_g, D, *V_g, 2);
    
    
    //Plotting parameters for surfPlot
    double a[3] = { -10.0, -10.0, 0.0};
    double b[3] = { 10.0, 10.0, 0.0};
    Plot<3> plt(100, a, b);
   
    plt.surfPlot(dx_epsilon, "dx_epsilon");
    
    mult(dx_eps_dx_V, 1.0, dx_epsilon, dx_V_g);
    mult(dy_eps_dy_V, 1.0, dy_epsilon, dy_V_g);
    mult(dz_eps_dz_V, 1.0, dz_epsilon, dz_V_g);
    
    gradeps_gradV.push_back(1.0, &dx_eps_dx_V);
    gradeps_gradV.push_back(1.0, &dy_eps_dy_V);
    gradeps_gradV.push_back(1.0, &dz_eps_dz_V);

    FunctionTree<3> temp_func(*MRA);
    add(temp_func, gradeps_gradV);
    temp_func *= 1.0/(4.0*pi);  
    gradeps_gradV.clear();
    
    FunctionTree<3> gamma(*MRA);
    mult(gamma, 1.0, temp_func, epsilon_inv);
  
    double integral = V->dot(gamma);
    
    return integral;
};

void testSCFCavity(){
    //Plotting parameters for surfPlot
    double a[3] = { -10.0, -10.0, 0.0};
    double b[3] = { 10.0, 10.0, 0.0};
    Plot<3> plt(10, a, b);
        
    // Precision parameters
    int max_scale = MRA->getMaxScale();
    double prec = Input.get<double>("rel_prec");
    double scf_prec = prec;
    
    // Initializing projector
    GridGenerator<3> grid(max_scale);
    MWProjector<3> project(prec, max_scale);
  
    //Importing molecular information.
    std::vector<std::string> mol_coords = Input.getData("Molecule.coords");
    Molecule mol(mol_coords);
    mol.printGeometry();

    // Orbtial energies
    double energy_n = -0.5;
    double energy_np1 = 0.0;
    double d_energy_n = 0.0;

    //atomic information from input file 
    Nuclei &nucs = mol.getNuclei();

    // Wave function
    
    FunctionTree<3> *phi_n = initialWaveFunction(prec, nucs);
    plt.surfPlot(*phi_n, "phi_n");
    // FunctionTree<3> *phi_n = new FunctionTree<3>(*MRA);
   // phi_n->setZero();
    FunctionTree<3> *phi_np1 = 0;
    
    //nuclear potential
    FunctionTree<3> *V_nuc = nuclearPotential(prec, nucs);
    plt.surfPlot(*V_nuc, "V_nuc_initial");
   

    FunctionTree<3> *rho_nuc = nuclearChargeDensity(prec, nucs);
    
    
    //cavity
    FunctionTree<3> eps(*MRA);
    FunctionTree<3> eps_inv(*MRA);
    {
        Timer timer;
        int oldlevel = TelePrompter::setPrintLevel(10);
        TelePrompter::printHeader(0, "Projecting cavityfunction and cavityfunction inverse");

        const double slope = Input.get<double>("Cavity.slope");
        //double slope = ; //slope of cavity, lower double --> steeper slope.
        double eps_0 = 1.0;
        
        const double eps_inf = Input.get<double>("Cavity.epsilon_inf");
        //double eps_inf = 2.0;

        
        //cavity
        CavityFunction cavity(nucs, slope, false, eps_0, eps_inf);
        project(eps, cavity);
    
        //cavity invers
        CavityFunction cavity_inv(nucs, slope, true, eps_0, eps_inf);
        project(eps_inv, cavity_inv);

        timer.stop();
        TelePrompter::printFooter(0, timer, 2);
        TelePrompter::setPrintLevel(oldlevel);
    
    }
    //initializing operators
   
    //derivative
    double boundary1 = 0.0, boundary2 = 0.0;
    ABGVOperator<3> D(*MRA, boundary1, boundary2);
    MWDerivative<3> applyDerivative(max_scale); 
    
    // Poisson/Greens
    PoissonOperator P(*MRA, scf_prec);

    //arithmetics 
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

    //Initializing functions for later use
    FunctionTree<3> *V = new FunctionTree<3>(*MRA);
    FunctionTree<3> *V_el_n = new FunctionTree<3>(*MRA);//nuclearPotential(prec, nucs);//need to be something, first iteration it's V_el is written as the nuclear potential \sim 1/r
    FunctionTree<3> *V_r = new FunctionTree<3>(*MRA);
    V_el_n->setZero();
    V_r->setZero();
    
    //V_el_n->setZero();
    
    
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
    while (errorPhi > scf_prec*10) {
        Timer cycle_t;    
        if (errorPhi < scf_prec*100){
            while (errorV > scf_prec*10){
            
                //derivative of electrostatic potential
                FunctionTree<3> dx_V_el(*MRA);
                FunctionTree<3> dy_V_el(*MRA);
                FunctionTree<3> dz_V_el(*MRA);
                  
                applyDerivative(dx_V_el, D, *V_el_n, 0);
                applyDerivative(dy_V_el, D, *V_el_n, 1);
                applyDerivative(dz_V_el, D, *V_el_n, 2);
                      
                //creating rho_eff (rho/eps)
                FunctionTree<3> rho_el(*MRA);
                FunctionTree<3> rho(*MRA);
                mult(rho_el, 2.0, *phi_n, *phi_n);//For larger systems, need to fix
                grid(rho, *rho_nuc);
                add(rho, 1.0, rho_el, 1.0, *rho_nuc);

                FunctionTree<3> rho_eff(*MRA);
                mult(rho_eff, 1.0, rho, eps_inv);

                //creating gamma (grad_eps*grad_V)/(4pi*eps)
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
              
                //////////////////////////////////////////// 
               // FunctionTreeVector<3> TEST_VEC;
               // TEST_VEC.push_back(1.0, &gamma);
               // TEST_VEC.push_back(-1.0, &rho);
               // TEST_VEC.push_back(1.0, &rho_eff);
               // FunctionTree<3> TEST(*MRA);
               // add(TEST,TEST_VEC);
               // 
               // FunctionTree<3> G_TEST(*MRA);
               // apply(G_TEST, P, TEST);
               // 
               // FunctionTree<3> G_rho(*MRA);
               // apply(G_rho, P, rho);

               // //double INTEGRAL_rho_Gtest = rho.dot(G_TEST);
               // //double INTEGRAL_Grho_test = G_rho.dot(TEST);
               // 
               // Input.get<double>("Cavity.slope");
               // const double input_rad = Input.get<double>("Cavity.slope");
               // const double input_ = Input.get<double>("Cavity.slope");
               // 
               // std::cout <<"slope  :" << Input.get<double>("Cavity.slope") << std::endl;
               // std::cout <<"eps_inf:" << Input.get<double>("Cavity.epsilon_inf") <<std::endl;
               // std::cout <<"radius :" << Input.get<double>("Cavity.radius") << std::endl;

               // std::cout << "int rho * G(gamma - rho + rho_eff):" << rho.dot(G_TEST)   << std::endl;
               // std::cout << "int Grho * (gamma - rho + rho_eff):" << G_rho.dot(TEST)   << std::endl;
                std::cout << "int gamma                         :" << gamma.integrate() << std::endl;
               //// std::cout << "int rho                           :" << rho.integrate() << std::endl;
               //// std::cout << "int rho_nuc                       :" << rho_nuc->integrate() << std::endl;
               //// std::cout << "int rho_el                        :" << rho_el.integrate() << std::endl;
               // std::cout << "int Grho*gamma                    :" << G_rho.dot(gamma)<<std::endl;
               // 
               // TEST_VEC.clear();
                /////////////////////////////////////
                FunctionTree<3> sum_rhoeff_gamma(*MRA);
                add(sum_rhoeff_gamma, 1.0, rho_eff, 1.0, gamma);

                //applying greensfunction
                V_el_np1 = new FunctionTree<3>(*MRA);
                apply(*V_el_np1, P, sum_rhoeff_gamma);
               

                //calculationg V_r 
                FunctionTree<3> sum_rhoeff_gamma_neg_rho(*MRA);
                V_r = new FunctionTree<3>(*MRA);
                add(sum_rhoeff_gamma_neg_rho, 1.0, sum_rhoeff_gamma, -1.0, rho);
                apply(*V_r, P, sum_rhoeff_gamma_neg_rho);
                
                std::cout << "int V_r                                           :" << V_r->integrate() << std::endl;
                //preparing for next iteration
                FunctionTree<3> error_func(*MRA);
                add(error_func, 1.0, *V_el_n, -1.0, *V_el_np1);

                errorV = sqrt(error_func.getSquareNorm());
                
                delete V_el_n;
                V_el_n = V_el_np1; 
               
                
                cycle += 1;
                std::cout << cycle << ":  " << errorV << std::endl; 
                

                //Plotting and testing
    //            plt.surfPlot(*phi_n, "phi_n");
  //              plt.surfPlot(*V_el_n, "V_el_n");
//                plt.surfPlot(gamma, "gamma");
//                plt.surfPlot(rho,"rho");
//                plt.surfPlot(rho_eff,"rho_eff");
//                plt.surfPlot(sum_rhoeff_gamma, "sum_rhoeff_gamma");
                // plt.surfPlot(dx_V_el, "dx_V_el"); 
               // plt.surfPlot(dy_V_el, "dy_V_el"); 
               // plt.surfPlot(dz_V_el, "dz_V_el"); 
               // plt.surfPlot(sum_rhoeff_gamma, "sum_rhoeff_gamma");
               // plt.surfPlot(*V, "V");
         //       plt.surfPlot(eps,"eps"); 
               // plt.surfPlot(*V_nuc,"V_nuc");
               // plt.surfPlot(rho_eff, "rho_eff");
               // plt.surfPlot(dx_eps, "dx_eps"); 
               //
    //            double integral_el_el = surfaceChargeTesting(eps, eps_inv, dx_eps, dy_eps, dz_eps, V_el_n, V_el_n);
    //            double integral_el_N = surfaceChargeTesting(eps, eps_inv, dx_eps, dy_eps, dz_eps, V_el_n, V_nuc);
    //            double integral_N_el = surfaceChargeTesting(eps, eps_inv, dx_eps, dy_eps, dz_eps, V_nuc, V_el_n);
    //            double integral_N_N = surfaceChargeTesting(eps, eps_inv, dx_eps, dy_eps, dz_eps, V_nuc, V_nuc);
    //            std::cout << "##########################" << integral_el_el << std::endl;
    //            std::cout << "##########################" << integral_el_N << std::endl;
    //            std::cout << "##########################" << integral_N_el << std::endl;
    //            std::cout << "##########################" << integral_N_N << std::endl;
            }
        
        }
      //not included for the one electron case
      
        FunctionTree<3> V_el(*MRA);
        FunctionTree<3> rho_el(*MRA);
        mult(rho_el, 2.0, *phi_n, *phi_n);
        apply(V_el, P, rho_el);
      

        FunctionTreeVector<3> V_sum;
        V_sum.push_back(1.0, V_r);
        V_sum.push_back(1.0, V_nuc);
        V_sum.push_back(.5, &V_el);//not included for the one electron case
        
        V = new FunctionTree<3>(*MRA);
        add(*V, V_sum); 

        plt.surfPlot(*V_el_n, "V_el_n");
        plt.surfPlot(*V,"V");
        
       //add(*V, 1.0, *V_el_n, 1.0, *V_nuc);


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
        grid(d_phi_n, *phi_np1);                     
        add(d_phi_n, 1.0, *phi_np1, -1.0, *phi_n); 
        
        errorPhi = sqrt(d_phi_n.getSquareNorm());
        errorV = 1;
        
        // Compute energy update
       // d_energy_n = Vphi.dot(d_phi_n)/phi_np1->getSquareNorm();
       // energy_np1 = energy_n + d_energy_n;
        


        //Computing energy directly
        //kinetic energy
        FunctionTree<3> dxphi(*MRA);
        FunctionTree<3> dyphi(*MRA);
        FunctionTree<3> dzphi(*MRA);
        
        applyDerivative(dxphi, D, *phi_np1, 0);
        applyDerivative(dyphi, D, *phi_np1, 1);
        applyDerivative(dzphi, D, *phi_np1, 2);

        double dxphi_dxphi = dxphi.dot(dxphi);
        double dyphi_dyphi = dyphi.dot(dyphi);
        double dzphi_dzphi = dzphi.dot(dzphi);
        
        double gradphi_gradphi = dxphi_dxphi + dyphi_dyphi + dzphi_dzphi; 

        double energy_T = 0.5*gradphi_gradphi/(phi_np1->getSquareNorm());
        
        //potential energy
        FunctionTree<3> Vphi_np1(*MRA);
        mult(Vphi_np1, 1.0, *V, *phi_np1);   
         
        double energy_V = Vphi_np1.dot(*phi_np1)/(phi_np1->getSquareNorm());
        
        energy_np1 = energy_V + energy_T;

        
        
        //nuc energy
        FunctionTree<3> V_nuc_phi_np1(*MRA);
        mult(V_nuc_phi_np1, 1.0, *V_nuc, *phi_np1);
        double energy_V_nuc = V_nuc_phi_np1.dot(*phi_np1)/(phi_np1->getSquareNorm());
        
        FunctionTree<3> V_r_phi_np1(*MRA);
        mult(V_r_phi_np1, 1.0, *V_r, *phi_np1);
        double energy_V_r = V_r_phi_np1.dot(*phi_np1)/(phi_np1->getSquareNorm());
        
        
        FunctionTree<3> V_el_phi_np1(*MRA);
        mult(V_el_phi_np1, 1.0, V_el, *phi_np1);
        double energy_V_el = 0.5*V_el_phi_np1.dot(*phi_np1)/(phi_np1->getSquareNorm());
        
        std::cout << "E_nuc     :"      << energy_V_nuc << std::endl;
        std::cout << "E_r       :"      << energy_V_r << std::endl;
        std::cout << "E_el      :"      << energy_V_el << std::endl;



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
        delete V; 

        //PLOTTING AND TESTING
        plt.surfPlot(*phi_n, "phi_n");

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

