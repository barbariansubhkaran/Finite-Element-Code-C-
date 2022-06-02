#include "fem.h"

using namespace std;
using namespace Eigen;




struct Element
{
	void CalculateStiffnessMatrix(const Matrix3f& D, vector<Triplet<float> >& triplets);

	Matrix<float, 3, 6> B;
	int nodesIds[3];
};

struct Constraint
{
	enum Type
	{
		UX = 1 << 0,
		UY = 1 << 1,
		UXY = UX | UY
	};
	int node;
	Type type;
};

int							nodesCount;
int							loadsCount;
VectorXf				nodesX;
VectorXf				nodesY;
VectorXf				loads;
vector< Element >		elements;
vector< Constraint >	constraints;

void Element::CalculateStiffnessMatrix(const Matrix3f& D, vector<Triplet<float> >& triplets)
{
	Vector3f x, y;
	x << nodesX[nodesIds[0]], nodesX[nodesIds[1]], nodesX[nodesIds[2]];
	y << nodesY[nodesIds[0]], nodesY[nodesIds[1]], nodesY[nodesIds[2]];
	
	Matrix3f C;
	C << Vector3f(1.0f, 1.0f, 1.0f), x, y;
	
	Matrix3f IC = C.inverse();

	for (int i = 0; i < 3; i++)
	{
		B(0, 2 * i + 0) = IC(1, i);
		B(0, 2 * i + 1) = 0.0f;
		B(1, 2 * i + 0) = 0.0f;
		B(1, 2 * i + 1) = IC(2, i);
		B(2, 2 * i + 0) = IC(2, i);
		B(2, 2 * i + 1) = IC(1, i);
	}
	Matrix<float, 6, 6> K = B.transpose() * D * B * C.determinant() / 2.0f;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Triplet<float> trplt11(2 * nodesIds[i] + 0, 2 * nodesIds[j] + 0, K(2 * i + 0, 2 * j + 0));
			Triplet<float> trplt12(2 * nodesIds[i] + 0, 2 * nodesIds[j] + 1, K(2 * i + 0, 2 * j + 1));
			Triplet<float> trplt21(2 * nodesIds[i] + 1, 2 * nodesIds[j] + 0, K(2 * i + 1, 2 * j + 0));
			Triplet<float> trplt22(2 * nodesIds[i] + 1, 2 * nodesIds[j] + 1, K(2 * i + 1, 2 * j + 1));

			triplets.push_back(trplt11);
			triplets.push_back(trplt12);
			triplets.push_back(trplt21);
			triplets.push_back(trplt22);
		}
	}
}

void SetConstraints(SparseMatrix<float>::InnerIterator& it, int index)
{
	if (it.row() == index || it.col() == index)
	{
		it.valueRef() = it.row() == it.col() ? 1.0f : 0.0f;
	}
}

void ApplyConstraints(SparseMatrix<float>& K, const vector<Constraint>& constraints)
{
	vector<int> indicesToConstraint;

	for (vector<Constraint>::const_iterator it = constraints.begin(); it != constraints.end(); ++it)
	{
		if (it->type & Constraint::UX)
		{
			indicesToConstraint.push_back(2 * it->node + 0);
		}
		if (it->type & Constraint::UY)
		{
			indicesToConstraint.push_back(2 * it->node + 1);
		}
	}

	for (int k = 0; k < K.outerSize(); ++k)
	{
		for (SparseMatrix<float>::InnerIterator it(K, k); it; ++it)
		{
			for (vector<int>::iterator idit = indicesToConstraint.begin(); idit != indicesToConstraint.end(); ++idit)
			{
				SetConstraints(it, *idit);
			}
		}
	}
}

int main(int argc, char *argv[])
{
	#pragma omp parallel num_threads(4)
	
	if ( argc != 3 )
    {
        cout<<"usage: "<< argv[0] <<" <input file> <output file>\n";
        return 1;
    }
		
	ifstream infile(argv[1]);
	ofstream outfile(argv[2]);
	
	float poissonRatio, youngModulus;
	infile >> poissonRatio >> youngModulus;
  
   //Elasticity matrix D

	Matrix3f D;
	D <<
		1.0f,			poissonRatio,	0.0f,
		poissonRatio,	1.0,			0.0f,
		0.0f,			0.0f,			(1.0f - poissonRatio) / 2.0f;

	D *= youngModulus / (1.0f - pow(poissonRatio, 2.0f));

	infile >> nodesCount;
	nodesX.resize(nodesCount);
	nodesY.resize(nodesCount);

	for (int i = 0; i < nodesCount; ++i)
	{
		infile >> nodesX[i] >> nodesY[i];
	}

	int elementCount;
	infile >> elementCount;

	for (int i = 0; i < elementCount; ++i)
	{
		Element element;
		infile >> element.nodesIds[0] >> element.nodesIds[1] >> element.nodesIds[2];
		elements.push_back(element);
	}

	int constraintCount;
	infile >> constraintCount;

	for (int i = 0; i < constraintCount; ++i)
	{
		Constraint constraint;
		int type;
		infile >> constraint.node >> type;
		constraint.type = static_cast<Constraint::Type>(type);
		constraints.push_back(constraint);
	}

	loads.resize(2 * nodesCount);
	loads.setZero();

	
	loads.resize(2 * nodesCount);
	loads.setZero();

	infile >> loadsCount;

	
	for (int i = 0; i < loadsCount; ++i)
	{
		int node;
		float x, y;
		infile >> node >> x >> y;
		loads[2 * node + 0] = x;
		loads[2 * node + 1] = y;
	}
	
	vector<Triplet<float> > triplets;
	for (vector<Element>::iterator it = elements.begin(); it != elements.end(); ++it)
	{
		it->CalculateStiffnessMatrix(D, triplets);
	}

	SparseMatrix<float> globalK(2 * nodesCount, 2 * nodesCount);
	globalK.setFromTriplets(triplets.begin(), triplets.end());

	ApplyConstraints(globalK, constraints);

	cout << "Global stiffness matrix:\n";
	cout <<  (globalK) << endl;

	cout << "Loads vector:" << endl << loads << endl << endl;

	SimplicialLDLT<SparseMatrix<float> > solver(globalK);

	VectorXf displacements = solver.solve(loads);
  
    cout << "Displacements vector:" << endl << displacements << endl;

	outfile << displacements << endl;
	cout << "Stresses:" << endl;

	for (vector<Element>::iterator it = elements.begin(); it != elements.end(); ++it)
	{
		#pragma omp for
		
		
		Matrix<float, 6, 1> delta;
		delta << displacements.segment<2>(2 * it->nodesIds[0]),
				 displacements.segment<2>(2 * it->nodesIds[1]),
				 displacements.segment<2>(2 * it->nodesIds[2]);

		Vector3f sigma = D * it->B * delta;
		float sigma_mises = sqrt(sigma[0] * sigma[0] - sigma[0] * sigma[1] + sigma[1] * sigma[1] + 3.0f * sigma[2] * sigma[2]);
   
      cout << sigma_mises << endl;

		outfile << sigma_mises << endl;
	}
	return 0;
}
