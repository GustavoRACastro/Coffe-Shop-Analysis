import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dados de vendas dos meses anteriores
meses = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  # 1 - Janeiro, 2 - Fevereiro, ..., 6 - Junho
vendas = np.array([81677.74, 76145.19, 98834.68, 118941.08, 156727.76, 166485.88])

# Criando e treinando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(meses, vendas)

# Prevendo as vendas para os próximos meses (julho a dezembro)
meses_futuros = np.array([7, 8, 9, 10, 11, 12]).reshape(-1, 1)
vendas_previstas = modelo.predict(meses_futuros)

# Exibindo os resultados
for mes, venda_prevista in zip(meses_futuros.flatten(), vendas_previstas):
    print(f"Mês {mes}: R${venda_prevista:.2f}")

# Plotando os dados e a linha de previsão
plt.scatter(meses, vendas, color='blue', label='Vendas Reais')
plt.plot(meses, modelo.predict(meses), color='red', label='Ajuste Linear')
plt.scatter(meses_futuros, vendas_previstas, color='green', label='Previsões Futuras')
plt.xlabel('Meses')
plt.ylabel('Valor das Vendas')
plt.title('Previsão de Vendas')
plt.legend()
plt.show()