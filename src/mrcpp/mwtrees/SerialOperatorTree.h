/**
*
*
*  \date Jul, 2016
*  \author Peter Wind <peter.wind@uit.no> \n
*  CTCC, University of Tromsø
*
*/

#ifndef SERIALOPERATORTREE_H_
#define SERIALOPERATORTREE_H_

#include <vector>

#include "SerialTree.h"
#include "parallel.h"

class OperatorTree;
class OperatorNode;

class SerialOperatorTree : public SerialTree<2> {
public:
    SerialOperatorTree(OperatorTree *tree, int max_nodes);
    virtual ~SerialOperatorTree();

    virtual void allocRoots(MWTree<2> &tree);
    virtual void allocChildren(MWNode<2> &parent);
    virtual void allocGenChildren(MWNode<2> &parent);

    virtual void deallocNodes(int serialIx);
    virtual void deallocGenNodes(int serialIx);

protected:
    OperatorNode *sNodes;       //serial OperatorNodes

    std::vector<OperatorNode*> nodeChunks;
    std::vector<double*> nodeCoeffChunks;

    char *cvptr_OperatorNode;   //virtual table pointer for OperatorNode
    OperatorNode* lastNode;     //pointer to the last active node

    OperatorNode* allocNodes(int nAlloc, int* serialIx, double **coefs_p);

private:
#ifdef HAVE_OPENMP
    omp_lock_t Soper_tree_lock;
#endif
};

#endif /* SERIALOPERATORTREE_H_*/
