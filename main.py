import imp
from EP1 import LUDecomposition, buildTridiagonalMatrix, solveLUSystem
from EP2 import integrateGauss
import numpy as np

from matplotlib import pyplot as plt

L = 1


def buildGridVector(n):
    return (np.linspace(0, L, num=(n+2)))


def buildDVector(gridVector, n, h, fx):
    """
    generate the result vector of the system of equations

    """
    dVector = np.zeros(n)
    for i in range(1, n+1):
        element = np.float64(0)
        element += integrateGauss(
            10,
            gridVector[i-1],
            gridVector[i],
            1,
            lambda y, x: fx(x)*(x-gridVector[i-1])/h
        )
        element += integrateGauss(
            10,
            gridVector[i],
            gridVector[i+1],
            1,
            lambda y, x: fx(x)*(gridVector[i+1]-x)/h
        )
        dVector[i-1] = element

    return dVector


def buildABCMatix(gridVector, n, h):
    aVector = np.zeros(n)
    bVector = np.zeros(n)
    cVector = np.zeros(n)

    # iterando sobre os nós
    for i in range(1, n):
        centerPoint = gridVector[i]
        previousPoint = gridVector[i-1]
        nextPoint = gridVector[i+1]

        # calculando o vetor b (diagonal principal)
        bVector[i-1] += integrateGauss(
            10,
            centerPoint,
            nextPoint,
            1,
            lambda y, x: (nextPoint-x)/h*(nextPoint-x)/h
        )
        bVector[i-1] += integrateGauss(
            10,
            previousPoint,
            centerPoint,
            1,
            lambda y, x: (x-previousPoint)/h*(x-previousPoint)/h
        )

        # calculando o produto interno entre o nó e o proximo, o vetor a e c é simehtrico
        symetricParameter = integrateGauss(
            10,
            centerPoint,
            nextPoint,
            1,
            lambda y, x: (x-gridVector[i])/h * (gridVector[i+1]-x)/h
        )

        aVector[i] = symetricParameter
        cVector[i-1] = symetricParameter

    # calculando o ultimo elmento de b  (phi_n *phi_n)
    i = n+1
    centerPoint = gridVector[i]
    previousPoint = gridVector[i-1]
    # nextPoint = gridVector[i+1]

    # bVector[i-1] += integrateGauss(
    #     10,
    #     centerPoint,
    #     nextPoint,
    #     1,
    #     lambda y, x: (nextPoint-x)/h*(nextPoint-x)/h
    # )
    # bVector[i-1] += integrateGauss(
    #     10,
    #     previousPoint,
    #     centerPoint,
    #     1,
    #     lambda y, x: (x-previousPoint)/h*(x-previousPoint)/h
    # )

    return (aVector, bVector, cVector)


def systemSolver(aVector, bVector, cVector, dVector):
    LVector, UVector = LUDecomposition(aVector, bVector, cVector)
    xVector = solveLUSystem(LVector, UVector, cVector, dVector)

    return xVector


def calculeTemperature(gridVector, alphaVector, n, h):
    # Criando array com um ponto extra entrecada noh para mostrar temperatura
    # temperatureGrid = np.zeros(2*n-1)

    # for i in range(1,n+1):
    #     print()

    temperatureGrid = np.zeros(n+2)

    for i in range(1, n+1):
        temperatureGrid[i] = alphaVector[i-1] * \
            ((gridVector[i]-gridVector[i-1])/h)

    return temperatureGrid


def bookMethod(xVec, n, h, px, fx, qx):
    # pre alocando vetores da matriz tridiagoanal e de resultado (d)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)


    for i in range(1,n+1):
        Q1 = ((1/h)**2)*(integrateGauss(
            10,
            xVec[i],
            xVec[i+1],
            1,
            lambda y, x: (xVec[i+1]-x)*(x-xVec[i])*qx(x)
        )
        )
        Q2 = ((1/h)**2)*(integrateGauss(
            10,
            xVec[i-1],
            xVec[i],
            1,
            lambda y, x: ((x-xVec[i-1])**2)*qx(x)
        )
        )
        Q3 = ((1/h)**2)*(integrateGauss(
            10,
            xVec[i],
            xVec[i+1],
            1,
            lambda y, x: ((xVec[i+1]-x)**2)*qx(x)
        )
        )
        Q4 = ((1/h)**2)*(integrateGauss(
            10,
            xVec[i-1],
            xVec[i],
            1,
            lambda y, x: px(x)
        )
        )
        Q5 = (1/h)*(integrateGauss(
            10,
            xVec[i-1],
            xVec[i],
            1,
            lambda y, x: (x-xVec[i-1])*fx(x)
        )
        )
        Q6 = (1/h)*(integrateGauss(
            10,
            xVec[i],
            xVec[i+1],
            1,
            lambda y, x: (xVec[i+1]-x)*fx(x)
        )
        )

        # printing elements of the approximation for analysis
        print(
        "{0:.2f}".format(Q1),
        "{0:.2f}".format(Q2),
        "{0:.2f}".format(Q3),
        "{0:.2f}".format(Q4),
        "{0:.2f}".format(Q5),
        "{0:.2f}".format(Q6),
        )

        b[i-1] += Q2 + Q3 + Q4
        if(i != 1):
            b[i-2] += Q4
            c[i-2] += -Q4
            a[i-1] += -Q4
        c[i-1] += Q1
        if(i != n):
            a[i] += Q1

        d[i-1] = Q5+Q6

    # para i=n
    i = n+1
    Q4 = ((1/h)**2)*(integrateGauss(
        10,
        xVec[i-1],
        xVec[i],
        1,
        lambda y, x: px(x)
    )
    )

    b[i-2] += Q4

    return(a,b,c,d)

def main():
    n = 7
    h = 1/(n+1)
    gridVector = buildGridVector(n)
    def fx(x): return 12*x*(1-x)-2

    aVector, bVector, cVector,dVector =bookMethod(gridVector, n, h, lambda x: 1, fx, lambda x: 1)
    print(1)
    # # montando matriz de elementos infinitos
    # dVector = buildDVector(gridVector, n, h, fx)
    # aVector, bVector, cVector = buildABCMatix(gridVector, n, h)

    aMatrix = buildTridiagonalMatrix(aVector, bVector, cVector)

    # resolvendo sistema
    alphaVector = systemSolver(aVector, bVector, cVector, dVector)

    temperatureVector = calculeTemperature(gridVector, alphaVector, n, h)

    plt.plot(gridVector, temperatureVector)
    plt.show()


main()
