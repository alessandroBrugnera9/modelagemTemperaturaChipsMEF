from cProfile import label
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
        if (np.abs(fx(gridVector[i])-temperatureVector[i]) > maxError):
            maxError = np.abs(fx(gridVector[i])-temperatureVector[i])

    quadraticError = np.sqrt(errorArr.sum()/len(errorArr))

    return quadraticError, maxError


def part1():
    # modelo mais simples com tamanho 1, k constante e funcao de calor determinada
    # avalia-se a variacao de n
    fig, ax = plt.subplots()
    L = 1
    for n in [7, 15, 31, 63, 127]:
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
            gridVector, temperatureVector, lambda x: ((1-x)**2)*(x**2))

        print("Erro Quadrático Médio para n={}, é: {:.3E}".format(n, quadraticError))
        print("Máximo Erro para n={}, é: {:.3E}".format(n, maxError))

        ax.plot(gridVector, temperatureVector,
                 label="n={}".format(n))

    # Plotando resultado exato
    exactResult = [((1-x)**2)*(x**2) for x in gridVector]
    ax.plot(gridVector, exactResult,
                label="Exato")

    ax.set_title("Simulação de Temperatura de Chip, por FEM em função de n")
    ax.set_xlabel("x [adm]")
    ax.set_ylabel("T [adm]")
    ax.legend()
    plt.show(block=False)


def part2():
    # modelo com k diferente de 1 porem constante, porem a geracao de calor varia como uma gaussiana
    L = 1
    n = 127
    h = L/(n+1)
    gridVector = buildGridVector(n, L)

    fig, ax = plt.subplots()
    
    # iterando com diferentes elementos da gaussiana para comparacao
    baseHeats = [10]
    sigmas = [0.1,1,10]
    for baseHeat in baseHeats:
        for sigma in sigmas:
            # modelando entrada e saida de calor
            def inHeat(x): return baseHeat*(np.exp(-((x-L/2)**2)/sigma**2))
            outHeat = 8
            def fx(x): return inHeat(x)-outHeat

            # montando matrizes de elementos infinitos
            # usando algoritmo de ritz como proposto no livro de Burden / Faires,
            # calcula-se as integrais utilizando o mehtodo da quadradutra de gauss de 10 pts
            # implementado no EP1
            # na modelagem do livro qx pode ser diferente de 0, mas na modelagem proposta
            # qx sempre serah 0, qx representa um multiplicador de T (sem derivar) na EDO
            aVector, bVector, cVector, dVector = ritzMethod(
                xVec=gridVector, n=n, h=h, kx=lambda x: 3.6, fx=fx, qx=lambda x: 0)

            # resolvendo sistema linear para calcular a contribuicao de cada noh
            alphaVector = systemSolver(aVector, bVector, cVector, dVector)

            # calculando o vetor final
            temperatureVector = calculeTemperature(gridVector, alphaVector, n, h)
            
            ax.plot(gridVector, temperatureVector,
                 label="\u03C3={}, Q0+={}".format(sigma,baseHeat))
    ax.set_title("Simulação de Temperatura de Chip, por FEM com calor gerado gaussiano")
    ax.set_xlabel("x [adm]")
    ax.set_ylabel("T [adm]")
    ax.legend()
    plt.show(block=False)

def part3():
    # modelo com k diferente de 1 porem constante, porem a geracao de calor varia como uma gaussiana
    # o resfriamento tambem varia
    L = 1
    n = 127
    h = L/(n+1)
    gridVector = buildGridVector(n, L)

    fig, ax = plt.subplots()
    
    # iterando com diferentes elementos da gaussiana para comparacao
    baseHeats = [1,10,30]
    baseHeatOut = 8
    sigmas = [0.1]
    theta = 2
    for baseHeat in baseHeats:
        for sigma in sigmas:
            # modelando entrada e saida de calor
            def inHeat(x): return baseHeat*(np.exp(-((x-L/2)**2)/sigma**2))
            def outHeat(x): return baseHeatOut*(np.exp(-((x)**2)/theta**2) +
                np.exp(-((x-L)**2)/theta**2))
            def fx(x): return inHeat(x)-outHeat(x)

            # montando matrizes de elementos infinitos
            # usando algoritmo de ritz como proposto no livro de Burden / Faires,
            # calcula-se as integrais utilizando o mehtodo da quadradutra de gauss de 10 pts
            # implementado no EP1
            # na modelagem do livro qx pode ser diferente de 0, mas na modelagem proposta
            # qx sempre serah 0, qx representa um multiplicador de T (sem derivar) na EDO
            aVector, bVector, cVector, dVector = ritzMethod(
                xVec=gridVector, n=n, h=h, kx=lambda x: 3.6, fx=fx, qx=lambda x: 0)

            # resolvendo sistema linear para calcular a contribuicao de cada noh
            alphaVector = systemSolver(aVector, bVector, cVector, dVector)

            # calculando o vetor final
            temperatureVector = calculeTemperature(gridVector, alphaVector, n, h)
            
            ax.plot(gridVector, temperatureVector,
                 label="\u03C3={}, Q+={}".format(sigma,baseHeat))
    ax.set_title("Simulação de Temperatura de Chip, por FEM com calor gerado e removido gaussiano")
    ax.set_xlabel("x [adm]")
    ax.set_ylabel("T [adm]")
    ax.legend()
    plt.show(block=False)


def part4():
    L = 1
    n=127
    fig, ax = plt.subplots()

    for d in [L/10, L/3, L/2 ]:
        h = L/(n+1)
        gridVector = buildGridVector(n, L)

        # modelando entrada e saida de calor
        baseHeat = 100
        sigma = 10
        def inHeat(x): return baseHeat*(np.exp(-((x-L/2)**2)/sigma**2))
        outHeat = 12
        def fx(x): return inHeat(x)-outHeat

        # modelando variacao no coecificiente de difusao de calor

        def kx(x):
            ks = 3.6
            ka = 60
            if (x > (L/2-d) and x < (L/2+d)):
                return ks
            else:
                return ka

        # montando matrizes de elementos infinitos
        # usando algoritmo de ritz como proposto no livro de Burden / Faires,
        # calcula-se as integrais utilizando o mehtodo da quadradutra de gauss de 10 pts
        # implementado no EP1
        # na modelagem do livro qx pode ser diferente de 0, mas na modelagem proposta
        # qx sempre serah 0, qx representa um multiplicador de T (sem derivar) na EDO
        aVector, bVector, cVector, dVector = ritzMethod(
            xVec=gridVector, n=n, h=h, kx=kx, fx=fx, qx=lambda x: 0)

        # resolvendo sistema linear para calcular a contribuicao de cada noh
        alphaVector = systemSolver(aVector, bVector, cVector, dVector)

        # calculando o vetor final
        temperatureVector = calculeTemperature(gridVector, alphaVector, n, h)

        ax.plot(gridVector, temperatureVector,
                label="d={}".format(d))
        ax.set_title("Simulação de Temperatura de Chip variando o tamanho do chip de sílicio")
        ax.set_xlabel("x [adm]")
        ax.set_ylabel("T [adm]")
        ax.legend()
        plt.show(block=False)


def main():
    part1()
    part2()
    part3()
    part4()


main()
