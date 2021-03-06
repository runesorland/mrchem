#ifndef SCF_H
#define SCF_H

#pragma GCC system_header
#include <Eigen/Core>

#include <vector>

#include "OrbitalAdder.h"
#include "TelePrompter.h"

class HelmholtzOperatorSet;
class Accelerator;
class FockOperator;
class OrbitalVector;
class Orbital;

class SCF {
public:
    SCF(HelmholtzOperatorSet &h);
    virtual ~SCF();

    virtual bool optimize() = 0;

    double getOrbitalPrecision() const { return this->orbPrec[0]; }
    double getOrbitalThreshold() const { return this->orbThrs; }
    double getPropertyThreshold() const { return this->propThrs; }

    void setThreshold(double orb_thrs, double prop_thrs);
    void setOrbitalPrec(double init, double final);
    void setMaxIterations(int m_iter) { this->maxIter = m_iter; }

    void setRotation(int iter) { this->rotation = iter; }
    void setCanonical(bool can) { this->canonical = can; }

protected:
    int maxIter;
    int rotation;    ///< number of iterations between localization/diagonalization
    bool canonical;  ///< use localized or canonical orbitals
    double orbThrs;  ///< Convergence threshold orbital update norm
    double propThrs; ///< Convergence threshold property
    double orbPrec[3];

    std::vector<double> orbError;
    std::vector<double> property;

    OrbitalAdder add;
    HelmholtzOperatorSet *helmholtz;// Pointer to external object, do not delete!

    bool checkConvergence(double err_o, double err_p) const;
    bool needLocalization(int nIter) const;
    bool needDiagonalization(int nIter) const;

    void adjustPrecision(double error);
    void resetPrecision();

    void printUpdate(const std::string &name, double P, double dP) const;
    double getUpdate(const std::vector<double> &vec, int i, bool absPrec) const;

    void printOrbitals(const Eigen::VectorXd &epsilon, const OrbitalVector &phi, int flag) const;
    void printConvergence(bool converged) const;
    void printCycle(int nIter) const;
    void printTimer(double t) const;
    void printMatrix(int level, const Eigen::MatrixXd &M,
                     const char &name, int pr = 5) const;

    void applyHelmholtzOperators(OrbitalVector &phi_np1,
                                 Eigen::MatrixXd &F_n,
                                 OrbitalVector &phi_n,
                                 bool adjoint = false);

    void applyHelmholtzOperators_P(OrbitalVector &phi_np1,
                                 Eigen::MatrixXd &F_n,
                                 OrbitalVector &phi_n,
                                 bool adjoint = false);

    virtual Orbital* getHelmholtzArgument(int i,
                                          Eigen::MatrixXd &F,
                                          OrbitalVector &phi,
                                          bool adjoint) = 0;
    virtual Orbital* getHelmholtzArgument_1(Orbital &phi_i) = 0;
    virtual Orbital* getHelmholtzArgument_2(int i,
					  int* OrbsIx,
                                          Eigen::MatrixXd &F,
                                          OrbitalVector &phi,
					  Orbital* part_1,
					  double coef_part1,
					  Orbital &phi_i,
                                          bool adjoint) = 0;

    Orbital* calcMatrixPart(int i,
                            Eigen::MatrixXd &M,
                            OrbitalVector &phi);
    Orbital* calcMatrixPart_P(int i,
                            Eigen::MatrixXd &M,
                            OrbitalVector &phi);
};

#endif // SCF_H

