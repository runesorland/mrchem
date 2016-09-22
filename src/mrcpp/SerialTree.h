/**
*
*
*  \date Jul, 2016
*  \author Peter Wind <peter.wind@uit.no> \n
*  CTCC, University of Tromsø
*
*/

#ifndef TREEALLOCATOR_H_
#define TREEALLOCATOR_H_

#include <Eigen/Core>
#include "parallel.h"


template<int D> class MultiResolutionAnalysis;
template<int D> class ProjectedNode;
template<int D> class GenNode;
template<int D> class MWNode;
template<int D> class MWTree;
template<int D> class FunctionTree;
template<int D> class FunctionNode;

template<int D>
class SerialTree  {
public:
  SerialTree(MWTree<D>* Tree, int max_nodes);
  //     SerialTree(const MultiResolutionAnalysis<D> &mra, int max_nodes);
    virtual ~SerialTree();

    FunctionTree<D>* getTree() { return static_cast<FunctionTree<D> *>(this->mwTree_p); }

    ProjectedNode<D>* allocNodes(int Nalloc, int* NodeIx);
    void DeAllocNodes(int NodeRank);
    GenNode<D>* allocGenNodes(int Nalloc, int* NodeIx);
    void DeAllocGenNodes(int NodeRank);
    double* allocCoeff(int NallocCoeff, MWNode<D>* node);
    void DeAllocCoeff(int DeallocIx);
    double** CoeffStack;
    double* allocGenCoeff(int NallocCoeff, MWNode<D>* node);
    void DeAllocGenCoeff(int DeallocIx);
    double** GenCoeffStack;
    void GenS_nodes(MWNode<D>* Node);
    void S_mwTransform(double* coeff_in, double* coeff_out, bool ReadOnlyScalingCoeff, int Children_Stride, bool overwrite=true);
    void S_mwTreeTransformUp();
    void S_mwTransformBack(double* coeff_in, double* coeff_out, int Children_Stride);

    void SerialTreeAdd(double c, FunctionTree<D>* &TreeB, FunctionTree<D>* &TreeC);
    void SerialTreeAdd_Up(double c, FunctionTree<D>* &TreeB, FunctionTree<D>* &TreeC);
    void RewritePointers();
    int* NodeStackStatus;
    int* GenNodeStackStatus;
    int* CoeffStackStatus;
    int* GenCoeffStackStatus;
    double* firstNodeCoeff;//pointer to the first node coefficents
    double* firstNode;//pointer to the first node
    
    Eigen::VectorXd* TempVector;

    friend class MWTree<D>;
    friend class ProjectedNode<D>;
    friend class MWNode<D>;
    friend class GenNode<D>;

    int nNodes;       //number of projected nodes already defined
    int nGenNodes;       //number of gen nodes already defined
    int nNodesCoeff;  //number of nodes Coeff already defined
    int nGenNodesCoeff;  //number of Gen nodes Coeff already defined

    double* SData; //Tree is defined as array of doubles, because C++ does not like void malloc
    double* SGenData; //Tree is defined as array of doubles, because C++ does not like void malloc

    //    const MWTree<D>* mwTree_p;
    MWTree<D>* mwTree_p;
    MultiResolutionAnalysis<D>* mra_p;
    ProjectedNode<D>* lastNode;//pointer to the last active node
    GenNode<D>* lastGenNode;//pointer to the last active Gen node
    double* lastNodeCoeff;//pointer to the last node coefficents
    double* lastGenNodeCoeff;//pointer to the last node coefficents
    int maxNodes;     //max number of nodes that can be defined
    int maxGenNodes;     //max number of Gen nodes that can be defined
    int maxNodesCoeff;//max number of nodes Coeff that can be defined
    int maxGenNodesCoeff;//max number of Gen nodes Coeff that can be defined
    int sizeTreeMeta; //The first part of the Tree is filled with metadata; reserved size:
    int sizeNodeMeta; //The first part of each Node is filled with metadata; reserved size:
    int sizeGenNodeMeta; //The first part of each Gen Node is filled with metadata; reserved size:
    int sizeNode;     //The dynamical part of the tree is filled with nodes metadata+coeff of size:
    int sizeGenNode;     //TGen nodes array is filled with Gen nodes metadata+coeff of size:
    int sizeNodeCoeff;//The dynamical part of the tree. Each Coeff set is of size:
    int sizeGenNodeCoeff;//The dynamical part of the tree. Each GenCoeff set is of size:
#ifdef HAVE_OPENMP
    omp_lock_t Stree_lock;
#endif
protected:
};

#endif /* TREEALLOCATOR_H_*/