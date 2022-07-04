import imp
from EP1 import LUDecomposition, solveLUSystem
from EP2 import integrateGauss
import numpy as np

L = 20E-3


def buildGridVector(n):
    return (np.linspace(0, L, num=(n+1)))


def buildDMatix(n, h, fx):
    """
    generate the result vector of the system of equations

    """
    dVector = np.zeros(n)
    for i in range(1, n):
        element = 0
        element += integrateGauss(
            
        )


def main():
    n = 7
    h = 1/(n+1)
    gridVector = buildGridVector(n)
    def fx(x): return 12*x*(1-x)-2

    matrixD = buildDMatix(gridVector, h, fx)


main()
