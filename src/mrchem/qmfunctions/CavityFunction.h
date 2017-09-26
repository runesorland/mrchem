#ifndef CAVITYFUNCTION_H
#define CAVITYFUNCTION_H

#include "RepresentableFunction.h"
#include "Nucleus.h"

class CavityFunction : public RepresentableFunction<3> {
public:
    CavityFunction(const Nuclei &nucs, double s, bool i = false, double eps_0 = 1.0, double eps_inf=10.0)
        : slope(s), nuclei(nucs), inverse(i), epsilon_0(eps_0), epsilon_inf(eps_inf) { }
    virtual ~CavityFunction() { }

    double get_epsilon_0(){
        return epsilon_0;
    }
    double get_epsilon_inf(){
        return epsilon_inf;
    }


    double evalf(const double *r) const {

        double c_tot = 1.0;//changes a lot


        double c_list[this->nuclei.size()];
        for (int i = 0; i < this->nuclei.size(); i++) {
            const Nucleus &nuc = this->nuclei[i];
            const double *coord = nuc.getCoord();
            double rad = Input.get<double>("Cavity.radius"); 
       //     double rad = nuc.getElement().getVdw();

            double s = sqrt(pow(coord[0]-r[0],2)+pow(coord[1]-r[1],2)+pow(coord[2]-r[2],2)) - rad;
            double theta = 0.5*(1+erf(s/slope));
            c_list[i] = 1-theta;
        }

        for (int i = 0; i < this->nuclei.size(); i++) {
            c_tot *= (1-c_list[i]);
        }
        c_tot = 1 - c_tot;

        double epsilon_r = epsilon_0 * exp(log(epsilon_inf/epsilon_0)*(1-c_tot));
        double epsilon_r_inv = (1/epsilon_0) * exp(log(epsilon_0/epsilon_inf)*(1-c_tot));

        if (inverse) { return epsilon_r_inv; }
        else { return epsilon_r; }
    }

protected:
    double slope;
    Nuclei nuclei;
    bool inverse;
    double epsilon_0;//inside cavity
    double epsilon_inf;//outside cavity
   
};




#endif // CAVITYFUNCTION_H
