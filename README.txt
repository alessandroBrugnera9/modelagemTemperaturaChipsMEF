O arquivo main.py executa as tarefas propostas no enunciado do EP.
A função main(), chama cada part (de 1 a 4) propondo modelagens cada vez mais complexas.
Cada parte plota um gráfico mostranado as variações dos parâmetros, na parte 1 também há um print dos valores de erro.
Os métodos principais são:
	ritzMethod - aplica o MEF para o grid e funções fornecidas ao método.
	calculeTemperature - utiliza os "alphas" calculados para calcular a solucao da EDO, no caso de temperatura.
	buildGridVector - discretiza a malha de 0 a L, com o n fornecido.

Além disso a integração é realizada pela função integrateGauss do EP2 e resolução dos sistemas tridiagonais é realizada pela função systemSolver do EP!.



Python 3.9.2
Numpy 1.21.4
Matplotlib 3.5.1