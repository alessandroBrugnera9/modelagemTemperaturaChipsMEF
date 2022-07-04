import os
from typing import Callable
from unittest import result
import numpy as np
from numpy import float64
from pyparsing import line


def createCoefficientsArray():
    f = open("dados.txt", "r")
    lines = f.readlines()
    nodes = np.zeros(3)
    weights = np.zeros(3)

    for i in range(3, 6):
        numbers = lines[i].split("\t")
        nodes[i-3] = float64(numbers[0])
        weights[i-3] = float64(numbers[1])

    try:
        os.mkdir('data')
    except:
        print("folder data already created")
    
    np.save('data/nodes6.npy', nodes)
    np.save('data/weights6.npy', weights)

    nodes = np.zeros(4)
    weights = np.zeros(4)

    for i in range(11, 15):
        numbers = lines[i].split("\t")
        nodes[i-11] = float64(numbers[0])
        weights[i-11] = float64(numbers[1])

    np.save('data/nodes8.npy', nodes)
    np.save('data/weights8.npy', weights)

    nodes = np.zeros(5)
    weights = np.zeros(5)

    for i in range(20, 25):
        numbers = lines[i].split("\t")
        nodes[i-20] = float64(numbers[0])
        weights[i-20] = float64(numbers[1])

    np.save('data/nodes10.npy', nodes)
    np.save('data/weights10.npy', weights)


def getCoefficients(n: int):
    nodes = np.load('data/nodes{}.npy'.format(n))
    weights = np.load('data/weights{}.npy'.format(n))

    return(nodes, weights)


def integrateGauss(n: int, a: float64, b: float64, fixedVariable: float64, mathematicalFunction: Callable[[float64, float64], float64]) -> float64:
    """
    calculate gauss quadratire using n elments from a to b of the function provided

    :param int n: number of coefficients for Gauss Quadrature
    :param float64 a: lower limit from the integral
    :param float64 b: upper limit from the integral
    :param float64 fixedVariable: fixed variable when integrating, when calculating double integral iteratively
    :param function mathematicalFunction: mathematical function to be integrated, the 1st arg is the fixed, 2nd is the node
    """
    nodes, weights = getCoefficients(n)

    result = float64(0)
    # calculate parts of gauss quadrature and applyting compensation for different limits
    for i in range(len(nodes)):
        node = ((b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*mathematicalFunction(
            fixedVariable,
            node)
        node = (-(b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*mathematicalFunction(
            fixedVariable,
            node)

    result *= (b-a)/2

    return result


def doubleIntegral(n: int, a: float64, b: float64,  c: Callable[[float64], float64], d: Callable[[float64], float64],mathematicalFunction: Callable[[float64, float64], float64]) -> float64:
    """
    calculate double integral iteratively using gauss quadratire, using n elments
    from a to b of the external integral, and c to d of the internal integral (these lmits can be dependent on the external variable(node))

    :param int n: number of coefficients for Gauss Quadrature
    :param float64 a: lower limit from the external integral 
    :param float64 b: upper limit from the external ntegral
    :param function c: upper limit from the internal integral, should be callable even when constant
    :param function d: upper limit from the internal integral, should be callable even when constant
    :param float64 fixedVariable: fixed variable when integrating, when calculating double integral iteratively
    :param function mathematicalFunction: mathematical function to be integrated, the 1st arg is the fixed, 2nd is the node
    """
    nodes, weights = getCoefficients(n)

    result = float64(0)
    # calculate parts of gauss quadrature and applyting compensation for different limits
    for i in range(len(nodes)):
        node = ((b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*integrateGauss(
            n,
            c(node),
            d(node),
            node,
            mathematicalFunction
        )
        node = (-(b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*integrateGauss(
            n,
            c(node),
            d(node),
            node,
            mathematicalFunction
        )

    result *= (b-a)/2

    return result


def example11(n: int) -> float64:
    return doubleIntegral(
        n,
        0,
        1,
        lambda x: 0,
        lambda x: 1,
        lambda x,y: (1)
    )

def example12(n: int) -> float64:
    return doubleIntegral(
        n,
        0,
        1,
        lambda x: 0,
        lambda x: 1-x,
        lambda x,y: (1-x-y)
    )


def example21(n: int) -> float64:
    return doubleIntegral(
        n,
        0,
        1,
        lambda x: 0,
        lambda x: 1-x**2,
        lambda x, y: 1
    )

def example22(n: int) -> float64:
    return doubleIntegral(
        n,
        0,
        1,
        lambda y: 0,
        lambda y: np.sqrt(1-y),
        lambda y, x: 1
    )

def example31(n: int) -> float64:
    return doubleIntegral(
        n,
        0.1,
        0.5,
        lambda x: x**3,
        lambda x: x**2,
        lambda x, y: np.sqrt(-(y*np.exp(y/x)/(x**2))**2+(np.exp(y/x)/x)+1),
    )

def example32(n: int) -> float64:
    return doubleIntegral(
        n,
        0.1,
        0.5,
        lambda x: x**3,
        lambda x: x**2,
        lambda x,y: np.exp(y/x),
    )

def example41(n: int) -> float64:
    return doubleIntegral(
        n,
        (1-1/4),
        1,
        lambda y: 0,
        lambda y: np.sqrt(1-y**2),
        lambda y, x: 2*np.pi*x,
    )

def example42(n: int) -> float64:
    return doubleIntegral(
        n,
        -1,
        1,
        lambda y: 0,
        lambda y: np.exp(-(y**2)),
        lambda y, x: 2*np.pi*x,
    )

def volumeSphericalCap(r,h):
    v=(np.pi*h**2/3)*(3*r-h)
    return v

def main():
    print("Análise da Quadratura Gaussiana para Integrais Duplas")
    ("Printando informacoes de cada exemplo e depois a aproximacao usando o metodo numerico para n=6, n=8 e n=10 respectivamente.\n")

    print("\n")
    print("Exemplo 1-1: ")
    print("Volume do cubo com aresta 1: ")
    print("Valor exato: 1")
    print(example11(6))
    print(example11(8))
    print(example11(10))
    
    print("\n")
    print("Exemplo 1-2: ")
    print("Volume do tetraedro com pontos: (0,0,0), (1,0,0), (0,1,0), (0,0,1)")
    print("Valor exato: 1/6 or 1.666666")
    print(example12(6))
    print(example12(8))
    print(example12(10))
    
    print("\n")
    print("Exemplo 2: ")
    print("Comparacao de integral dupla utilzando ordem dydx ou dxdy")
    print("Valor exato: 2/3 or 0.6666667")
    print("dydx:")
    print(example21(6))
    print(example21(8))
    print(example21(10))
    print("dxdy:")
    print(example22(6))
    print(example22(8))
    print(example22(10))
    
    print("\n")
    print("Exemplo 3-1: ")
    print("Area da superficie z=e^(y/x), delimitado por 0.1<x<0.5 e x^3<y<x^2")
    print(example31(6))
    print(example31(8))
    print(example31(10))
    
    print("\n")
    print("Exemplo 3-2: ")
    print("Volume abaixo da superficie z=e^(y/x), delimitado por 0.1<x<0.5 e x^3<y<x^2")
    print(example42(6))
    print(example42(8))
    print(example42(10))

    print("\n")
    print("Exemplo 4-1: ")
    print("Volume da calota esferico com h=1/4 e r=1")
    print("Valor exato: ", volumeSphericalCap(1, 1/4))
    print(example41(6))
    print(example41(8))
    print(example41(10))
    
    print("Exemplo 4-2: ")
    print("Volume do solido de revoluao no eixo y delimitado pela regiao 0<x<e^(-y^2) e -1<y<1")
    print(example42(6))
    print(example42(8))
    print(example42(10))
    
createCoefficientsArray()
main()