from typing import Tuple
import numpy as np


def buildTestSystem(size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aVector = np.zeros(size)
    bVector = np.zeros(size)
    cVector = np.zeros(size)
    dVector = np.zeros(size)

    for i in range(size-1):
        upperValue = (2*(i+1)-1)/(4*(i+1))
        aVector[i] = 1 - upperValue
        bVector[i] = 2
        cVector[i] = upperValue
        dVector[i] = np.cos(2*np.pi*((i+1)**2)/(size**2))
    # overriding last entry
    upperValue = (2*size-1)/(4*size)
    aVector[size-1] = 1 - upperValue
    bVector[size-1] = 2
    cVector[size-1] = upperValue
    dVector[size-1] = 1

    return (aVector, bVector, cVector, dVector)


def LUDecomposition(aVector, bVector, cVector):
    # the tridigonalmatrix is represented using only 3 vectors
    # only the important value of L (upperdiagonal),U(maindiagional) are calculated
    size = len(aVector)

    LVector = np.zeros(size)
    UVector = np.zeros(size)
    UVector[0] = bVector[0]

    for i in range(1, size):
        LVector[i] = aVector[i]/UVector[i-1]
        UVector[i] = bVector[i]-LVector[i]*cVector[i-1]

    return (LVector, UVector)


def solveLUSystem(LVector, UVector, cVector, dVector) -> np.ndarray:
    # L,U,d,c,x,y are vectors necessay to solve the linear syste
    size = len(LVector)
    xVector = np.zeros(size)
    yVector = np.zeros(size)
    yVector[0] = dVector[0]

    for i in range(1, size):
        yVector[i] = dVector[i]-LVector[i]*yVector[i-1]
    xVector[size-1] = yVector[size-1]/UVector[size-1]
    for j in range(size-2, -1, -1):
        xVector[j] = (yVector[j]-cVector[j]*xVector[j+1])/UVector[j]

    return xVector


def buildTridiagonalMatrix(aVector, bVector, cVector, offset=0) -> np.ndarray:
    size = len(aVector)-offset
    aMatrix = np.zeros((size, size))
    for i in range(size-1):
        aMatrix[i][i] = bVector[i]

        # upper diagonal
        aMatrix[i][i+1] = cVector[i]

        # lower diagonal
        aMatrix[i][i-1] = aVector[i]

    aMatrix[size-1][size-1] = bVector[size-1]
    aMatrix[size-1][0] = cVector[size-1]
    aMatrix[size-1][size-1-1] = aVector[size-1]

    return aMatrix


def main(matrixSize=20, cyclic=True):
    print("Cyclic: ", cyclic)
    aVector, bVector, cVector, dVector = buildTestSystem(matrixSize)
    
    if cyclic:
        # creating vectors necessaries to solve cyclic systems
        # building v vector  for cyclic systems
        vVector = np.zeros(matrixSize-1)
        vVector[0] = aVector[0]
        vVector[-1] = cVector[-2]
        # building w vector  for cyclic systems
        wVector = np.zeros(matrixSize-1)
        wVector[0] = cVector[-1]
        wVector[-1] = aVector[-1]

        # Creating n-1xn-1 system from the main Matrix, and solving to find important relations
        LVector, UVector = LUDecomposition(aVector[:-1], bVector[:-1], cVector[:-1])
        yTilde = solveLUSystem(LVector, UVector, cVector[:-1], dVector[:-1])
        zTilde = solveLUSystem(LVector, UVector, cVector[:-1], vVector)

        xN = (dVector[-1]-cVector[-1]*yTilde[0]-aVector[-1]*yTilde[-1])/(bVector[-1]-cVector[-1]*zTilde[0]-aVector[-1]*zTilde[-1])
        xTilde = yTilde-xN*zTilde

        xVector = np.append(xTilde, xN)
    else: # getting the solution for non cyclic systems
        aVector[0]=0
        cVector[-1]=0
        LVector, UVector = LUDecomposition(aVector, bVector, cVector)
        xVector = solveLUSystem(LVector, UVector, cVector, dVector)

    aMatrix = buildTridiagonalMatrix(aVector, bVector, cVector)



    print("Solution: ")
    print(xVector.tolist())
    # np.savetxt("mydata.csv", xVector, delimiter=' & ', fmt='%2.1e', newline=' & ')
    print()
    # Analyzing the solutions
    calculatedValue = aMatrix@xVector
    print("Comparing true value of the system multiplication (D Vector) to A*x:")
    print((calculatedValue-dVector).tolist())
    print()
    print("Residual Quadratic Error: ", np.square(calculatedValue-dVector).sum())
    print("Mean Root Quadratic Error: ",np.sqrt(np.square(calculatedValue-dVector).mean()))


if __name__=='__main__':
    main(matrixSize=20, cyclic=False)
    main(matrixSize=20, cyclic=True)