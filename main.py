import imp
from EP1 import LUDecomposition, solveLUSystem
from EP2 import integrateGauss
import numpy as np

L = 20E-3


def buildGridVector(n):
    return (np.linspace(0, L, num=(n+1)))


def buildDMatix(gridVector, n, h, fx):
    """
    generate the result vector of the system of equations

    """
    dVector = np.zeros(n)
    for i in range(1, n):
        element = np.float64(0)
        element += integrateGauss(
            10,
            gridVector[i-1],
            gridVector[i],
            1,
            lambda y, x: (x-gridVector[i-1])/h
        )
        element += integrateGauss(
            10,
            gridVector[i],
            gridVector[i+1],
            1,
            lambda y, x: (gridVector[i+1]-x)/h
        )
        dVector[i-1] = element

    return dVector

def main():
    n = 7
    h = 1/(n+1)
    gridVector = buildGridVector(n)
    def fx(x): return 12*x*(1-x)-2

    matrixD = buildDMatix(gridVector, n, h, fx)


main()