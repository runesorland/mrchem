#pragma once
#include "MathUtils.h"
#include "RepresentableFunction.h"
#include "Nucleus.h"

class CavityFunction : public RepresentableFunction<3> {
public:
    CavityFunction(const Nuclei &nucs, double s, double eps_0 = 1.0, double eps_inf=10.0)
        : nuclei(nucs), slope(s), inverse(false), epsilon_0(eps_0), epsilon_inf(eps_inf) { }
    virtual ~CavityFunction() { }

    void setInverse(bool i) { this->inverse = i; }

    double getEpsilon_0() const { return epsilon_0; }
    double getEpsilon_inf() const { return epsilon_inf; }

    double evalf(const double *r) const {
        double c_list[this->nuclei.size()];
        for (int i = 0; i < this->nuclei.size(); i++) {
            const Nucleus &nuc = this->nuclei[i];
            const double *coord = nuc.getCoord();
            //double rad = Input.get<double>("Cavity.radius"); 
            double rad = nuc.getElement().getVdw();
            //println(0, "Radius " << rad);
            //double rad = 3.78;

            double s = MathUtils::calcDistance(3, coord, r) - rad;
            double theta = 0.5*(1.0 + erf(s/this->slope));
            c_list[i] = 1.0 - theta;
        }

        double c_tot = 1.0;
        for (int i = 0; i < this->nuclei.size(); i++) {
            c_tot *= (1 - c_list[i]);
        }
        c_tot = 1 - c_tot;

        double result = 0.0;
        if (this->inverse) {
            result = (1.0/this->epsilon_0)*exp(log(this->epsilon_0/this->epsilon_inf)*(1.0-c_tot));
        } else {
            result = this->epsilon_0*exp(log(this->epsilon_inf/this->epsilon_0)*(1.0-c_tot));
        }
        return result;
    }

protected:
    Nuclei nuclei;
    double slope;
    bool inverse;
    double epsilon_0;   //inside cavity
    double epsilon_inf; //outside cavity
};

