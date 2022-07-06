import imp
from EP1 import LUDecomposition, buildTridiagonalMatrix, solveLUSystem
from EP2 import integrateGauss
import numpy as np

from matplotlib import pyplot as plt


def buildGridVector(n, L):
    return (np.linspace(0, L, num=(n+2)))


def systemSolver(aVector, bVector, cVector, dVector):
    LVector, UVector = LUDecomposition(aVector, bVector, cVector)
    xVector = solveLUSystem(LVector, UVector, cVector, dVector)

    return xVector


def calculeTemperature(gridVector, alphaVector, n, h):
    # Criando array com tamanho n+1 com as temperaturas

    temperatureGrid = np.zeros(n+2)

    for i in range(1, n+1):
        temperatureGrid[i] = alphaVector[i-1] * \
            ((gridVector[i]-gridVector[i-1])/h)

    return temperatureGrid


def ritzMethod(xVec, n, h, kx, fx, qx):
    # pre alocando vetores da matriz tridiagoanal e de resultado (d)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for i in range(1, n+1):
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
            lambda y, x: kx(x)
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
        # print(
        # "{0:.2f}".format(Q1),
        # "{0:.2f}".format(Q2),
        # "{0:.2f}".format(Q3),
        # "{0:.2f}".format(Q4),
        # "{0:.2f}".format(Q5),
        # "{0:.2f}".format(Q6),
        # )

        b[i-1] += Q2 + Q3 + Q4
        if(i != 1):
            b[i-2] += Q4
            c[i-2] += -Q4
            a[i-1] += -Q4
        c[i-1] += Q1
        if(i != n):
            a[i] += Q1

        d[i-1] = Q5+Q6

    # para i=n+1, para o ultimo elmento da diagonal prinicpal
    i = n+1
    Q4 = ((1/h)**2)*(integrateGauss(
        10,
        xVec[i-1],
        xVec[i],
        1,
        lambda y, x: kx(x)
    )
    )

    b[i-2] += Q4

    return(a, b, c, d)


def calculateErrors(gridVector, temperatureVector, fx):
    errorArr = np.zeros_like(temperatureVector)
    maxError = np.float64(0)

    for i in range(len(gridVector)):
        errorArr[i] = (fx(gridVector[i])-temperatureVector[i])**2
        if (np.abs(fx(gridVector[i])-temperatureVector[i])>maxError):
            maxError = np.abs(fx(gridVector[i])-temperatureVector[i])

    quadraticError = np.sqrt(errorArr.sum()/len(errorArr))

    return quadraticError, maxError


def part1():
    L = 1
    for n in [7, 15, 31, 63, 127, 511]:
        h = L/(n+1)
        gridVector = buildGridVector(n, L)
        def fx(x): return 12*x*(1-x)-2
        
        # montando matrizes de elementos infinitos
        # usando algoritmo de ritz como proposto no livro de Burden / Faires,
        # calcula-se as integrais utilizando o mehtodo da quadradutra de gauss de 10 pts
        # implementado no EP1
        # na modelagem do livro qx pode ser diferente de 0, mas na modelagem proposta
        # qx sempre serah 0, qx representa um multiplicador de T (sem derivar) na EDO
        aVector, bVector, cVector, dVector = ritzMethod(
            xVec=gridVector, n=n, h=h, kx=lambda x: 1, fx=fx, qx=lambda x: 0)

        # resolvendo sistema linear para calcular a contribuicao de cada noh
        alphaVector = systemSolver(aVector, bVector, cVector, dVector)

        # calculando o vetor final
        temperatureVector = calculeTemperature(gridVector, alphaVector, n, h)

        quadraticError, maxError = calculateErrors(
            gridVector, temperatureVector, fx)

        print("Erro Quadrático Médio para n={}, é: {:.3f}".format(n, quadraticError))
        print("Máximo Erro para n={}, é: {:.3f}".format(n, maxError))

        plt.ion()
        plt.plot(gridVector, temperatureVector)
        # plt.show()
        print(1)

def part2():
    L = 1
    for n in [7]:
        h = L/(n+1)
        gridVector = buildGridVector(n, L)
        def fx(x): return 12*x*(1-x)-2

        # montando matrizes de elementos infinitos
        # usando algoritmo de ritz como proposto no livro de Burden / Faires,
        # calcula-se as integrais utilizando o mehtodo da quadradutra de gauss de 10 pts
        # implementado no EP1
        # na modelagem do livro qx pode ser diferente de 0, mas na modelagem proposta
        # qx sempre serah 0, qx representa um multiplicador de T (sem derivar) na EDO
        aVector, bVector, cVector, dVector = ritzMethod(
            xVec=gridVector, n=n, h=h, kx=lambda x: 3.6, fx=fx, qx=lambda x: 1)

        # resolvendo sistema linear para calcular a contribuicao de cada noh
        alphaVector = systemSolver(aVector, bVector, cVector, dVector)

        # calculando o vetor final
        temperatureVector = calculeTemperature(gridVector, alphaVector, n, h)

        plt.plot(gridVector, temperatureVector)
        plt.show()
        print(1)


def main():
    part1()

    # # plotando a temperatura ao longo do eixo
    # plt.plot(gridVector, temperatureVector)
    # plt.show()
    # print(1)


main()
