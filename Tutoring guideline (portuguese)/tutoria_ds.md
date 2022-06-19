# Uma sugestão de estudos para mentoria em Data Science

## Pré-requisitos esperados do mentorado

- **Probabilidade**: A linguagem de Aprendizado de Máquina é intrinsicamente probabilística. É necessário que o mentorado esteja confortável com os conceitos básicos de probabilidade (no mínimo em um [nível introdutório do assunto](https://gradmat.ufabc.edu.br/disciplinas/ipe/)). Isso significa, conhecer a definição (ingênua) de espaço de probabilidade, entender sobre probabilidade condiciona e independência, variáveis aleatórias discretas e contínuas, conhecer as principais distribuições de variáveis aleatórias e saber calcular esperanças, variâncias e correlações. Caso esse assunto seja intimidador ainda, é sugerido uma revisão detalhada dos tópicos seguindo, por exemplo, o curso [Introdução à Probabilidade e Estatístíca (IPE) do professor Rafael Grisi da UFABC](https://www.youtube.com/channel/UCKGYUbUHWSDr8pFaDd7JWCg/videos).
- **Lógica de programação**: Não é esperado que o mentorado já tenha utilizado Python, mas aguarda-se que já tenha experiência prévia com programação em outra linguagem e esteja confortável com as estruturas de repetição, condições e operadores lógicos, além de algumas estruturas de dados básicas como listas, vetores e strings. Se esse não for o caso do mentorado, o curso de [introdução à programação do Kaggle](https://www.kaggle.com/learn/intro-to-programming) pode ser um bom começo. Se ainda assim, for necessário gastar mais tempo vendo esses tópicos, aulas selecionadas da [playlist de Python do Curso em Vídeo](https://youtube.com/playlist?list=PLvE-ZAFRgX8hnECDn1v9HNTI71veL3oW0) podem ser uma boa referência.

## Ensinamentos do HDFS: redundância é importante

Durante esse guia de materiais de mentoria, por selecionar partes específicas de cursos diferentes, vários assuntos podem ser vistos em repetição com uma abordagem ligeiramente diferente. Isso é intencional. Acredito que para formentar essa base da forma mais robusta possível, é importante que os assuntos realmente estejam absorvidos e vê-los algumas vezes reforça isso.  

## 0) Como usar esse guia de atividades de tutoria?

### 0.1) Requisitos sugeridos

Os pré-requisitos são para organização caso queira seguir outra ordem ou pular etapas.

### 0.2) Motivação

A motivação é para o tutor saber porque aquela atividade faz sentido naquele momento (pode ser compartilhado com o mentorado se achar que faz sentido ou ele perguntar). 

### 0.3) Descrição

A descrição é o texto sugerido para passar pro aluno na ferramenta utilizada para registrar as atividades. Eu sugiro usar o Trello: ele tem uma estrutura estilo Kanban que você pode organizar os cards, bem parecido com o como utilizamos o Jira.

# Atividades recorrentes

## A) HackerRank

### A.1) Requisitos sugeridos

Atividade 2.

### A.2) Motivação

O HackerRank tem várias playlists interessantes que te ensinam com exercícios sobre particularidades do Python. Você conhece estrutura de dados não triviais enquanto revisita as que você acha que já domina aprendendo formas novas de trabalhar com. Variações dessa ferramentas costumam ser utilizadas em processos seletivos.

### A.3) Descrição

Fazer pelo menos 1 hora de atividades propostas nas playlists de Python do [HackerRank](https://www.hackerrank.com/domains/python) toda semana. Rapidinho você já vai ter visto a maioria dos tipos relevantes e eles vão te ajudar a deixar seu código mais Pythonic. 

## B) Stratascratch

### B.1) Requisitos sugeridos

Atividade -.

### B.2) Motivação

Pandas é uma biblioteca que você só vai dominar se utilizar muito. A ideia aqui é ter alguns exercícios mais ou menos clássicos para estar se familiarizando com as funções clássicas.

### B.3) Descrição

Fazer pelo menos 0.5 horas de atividades propostas utilizando Pandas do [Stratascratch]([stratascratch.com](http://stratascratch.com/)) toda semana. Você consegue resolver em SQL também e é um lugar bacana para isso também.

# Atividades "Sequenciais"

## 1) Instalar Python (pelo Anaconda)

### 1.1) Requisitos sugeridos

Não há.

### 1.2) Motivação

A ideia é deixar o ambiente pronto para utilizar Jupyter Notebooks. São ótimas ferramentas para exploração e, na prática, é o ambiente mais adotado no dia-a-dia do cientista de dados. Durante os encontros é sempre útil mostrar os atalhos mais comuns e boas práticas (deixar o notebook organizado), mas falar todos agora só vai confundir o mentorando. Eventualmente, no final, podemos mostrar mais coisas de IDE, como VSCode etc, mas não acho útil no início.

### 1.3) Descrição

Seguir os passos descritos aqui:  [Instalando Python para Aprendizado de Máquina - LAMFO](https://lamfo-unb.github.io/2017/06/10/Instalando-Python/).
Não precisa instalar TensorFlow/Git por enquanto.

## 2) Nivelamento Python (Kaggle Learn)

### 2.1) Requisitos sugeridos

Atividade 1. Já assume conhecimento de programação básico em alguma linguagem.

### 2.2) Motivação

Python é A LINGUAGEM para DS. Muitas bibliotecas boas prontas e a maioria das empresas adota como padrão. O curso do Kaggle Learn é ótimo porque foca nas partes mais importantes pra quem tá migrando de linguagem e em coisas que serão úteis em breve para sklearn, pandas etc.

### 2.3) Descrição 

[Learn Python Tutorials](https://www.kaggle.com/learn/python). Tempo estimado de 5 horas.

## 3) Aula introdutória do curso (básico) do Andrew

### 3.1) Requisitos sugeridos

Não há.

### 3.2) Motivação

Aqui o didático Andrew apresenta os tipos de aprendizado e dá exemplos legais. É um bom ponto pra pedir pro mentorado dar exemplos de problemas que dá pra aplicar e também motivar algumas aplicações.

### 3.3) Descrição

- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
- Lecture 1.1 What Is Machine Learning (7 min)
- Lecture 1.2 — Supervised Learning (13 min)
- Lecture 1.3 — Unsupervised Learning (14 min)

## 4) Aula de Regressão Linear do curso (básico) do Andrew

### 4.1) Requisitos sugeridos

Atividade 3.

### 4.2) Motivação

Regressão Linear é o algoritmo mais simples possível de aprendizado de máquina. É uma oportunidade legal de mostrar que as ideias gerais são bem parecidas com esse exemplo (pelo menos pra problemas supervisionados): criar uma função que aproxima bem os dados.

### 4.3) Descrição

- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
- Lecture 2.1 — Linear Regression With One Variable | Model Representation (8:11)
- Lecture 2.2 — Linear Regression With One Variable | CostFunction (8:13)
- Lecture 2.3 — Linear Regression With One Variable | Cost Function Intuition (11:10)
- Lecture 2.4 — Linear Regression With One Variable | Cost Function Intuition #2 (8:49)
- Lecture 2.5 — Linear Regression With One Variable | Gradient Descent (11:31)
- Lecture 2.6 — Linear Regression With One Variable | Gradient Descent Intuition (11:52)
- Lecture 2.7 — Linear Regression With One Variable | Gradient Descent For Linear Regression (10:21)
- Lecture 2.8 — What's Next (5:50)

## 5) Introdução ao Numpy

### 5.1) Requisitos sugeridos

Atividade 2.

### 5.2) Motivação

Numpy é a principal ferramenta para mexer em matrizes no Python. Todas as outras bibliotecas são construídas em cima dela. É primordial entender o básico e eventualmente masterizar.

### 5.3) Descrição

- Numpy in 5 min: [Learn NUMPY in 5 minutes - BEST Python Library!](https://youtu.be/xECXZ3tyONo)(20 min)

- Se precisar ver mais, esse vídeo aqui também é interessante: [Complete Python NumPy Tutorial (Creating Arrays, Indexing, Math, Statistics, Reshaping)](https://youtu.be/GB9ByFAIAH4) (1 hora)

## 6) Implementing from ground up: Regressão Linear Simples

### 6.1) Requisitos sugeridos

Atividades 2, 3, 4 e 5.

### 6.2) Motivação

Essa atividade é uma forma de colocar em prática tudo que vimos até agora. 

### 6.3) Descrição

- Utilizando Python (principalmente o numpy), você deve construir uma função que recebe seu conjunto de dados X_train, y_train e retorna os pesos de uma regressão linear simples (ou seja, X_train é unidimensional), utilizando gradiente descendente para fazer esse cálculo, como visto no curso do Andrew. Pode criar o seu conjunto X_train, y_train como quiser, mas seu código deve ser robusto o suficiente para poder trocar os valores e continuar rodando da forma correta.
- Defina critérios de parada que você achar apropriado para o gradiente descendente.
- Em seguida, com os pesos calculados, você deverá fazer uma função que prevê os y para um conjunto X qualquer.
- Pode ser interessante utilizar algumas bibliotecas gráficas para visualizar o que você está fazendo. A mais famosa, e que eu mais gosto é o matplotlib. Nessa [playlist do Corey Schafer](https://youtube.com/playlist?list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_) temos demonstrações de vários gráficos úteis.

## 7) Curso de básico de ML do Kaggle Learn

### 7.1) Requisitos sugeridos

Atividades 2, 3 e 4.

### 7.2) Motivação

Tirando problemas específicos, o dia a dia do cientista de dados não é criando os modelos do zero. O scikit-learn é uma das mais robustas bibliotecas com dezenas de modelos já construídos seguindo os melhores padrões de desenvolvimento de software. Tem uma comunidade Open Source incrível que dá suporte e guia os desenvolvimentos dela. Esse curso do Kaggle é um primeiro contato com o scikit-learn. Pra começar a entender a ideia de fit/predict.

### 7.3) Descrição

[Learn Intro to Machine Learning Tutorials](https://www.kaggle.com/learn/intro-to-machine-learning). Tempo estimado de 3 horas.

## 8) Introdução à Programação Orientada à Objetos (POO)

### 8.1) Requisitos sugeridos

Atividade 7.

### 8.2) Motivação

A orientação orientada à objetos é o paradigma de programação principal do Python. A forma de abstração  que ele nos oferece é muito poderosa e permite construir códigos complexos de uma forma estruturada e reaproveitável, com manutenção facilitada. A ideia dessa atividade não é ficar um mestre em POO, mas conhecer por cima a ideia para saber que existe e entender que o scikit-learn e outras bibliotecas do Python utilizam ela. No futuro, esse tópico pode ser revisto, entendendo agora a utilização de heranças e boas práticas (como os princípios SOLID).  

### 8.3) Descrição

- [Python OOP Tutorial 1: Classes and Instances](https://youtu.be/ZDa-Z5JzLYM) (15 min)
- Extra: uma discussão sobre diferentes formas de se programar (paradigmas): [1.1 - Programação Funcional em Haskell: Paradigmas de Programação](https://youtu.be/CzGSaXbPFRA) (27 min) - O Python tem várias coisas bem úteis de programação funcional então é legal conhecer por cima as ideias também.

## 9) Aula de revisão de matrizes do curso (básico) do Andrew

### 9.1) Requisitos sugeridos

Atividade 4.

### 9.2) Motivação

Operações com matrizes são muito importantes para otimizar a computação de redes neurais. No curto prazo, vai facilitar a representação da regressão linear multivariada.

### 9.3) Descrição

- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
- Lecture 3.1 — Linear Algebra Review | Matrices And Vectors - 8:46
- Lecture 3.2 — Linear Algebra Review | Addition And Scalar Multiplication - 6:54
- Lecture 3.3 — Linear Algebra Review | Matrix Vector Multiplication - 13:40
- Lecture 3.4 — Linear Algebra Review | Matrix-Matrix Multiplication - 11:10
- Lecture 3.5 — Linear Algebra Review | Matrix Multiplication Properties - 9:03
- Lecture 3.6 — Linear Algebra Review | Inverse And Transpose - 11:14

## 10) Aula de Regressão Linear Multivariada (e polinomial) do curso (básico) do Andrew

### 10.1) Requisitos sugeridos

Atividade 9.

### 10.2) Motivação

Na vida real vamos utilizar dezenas, centenas, milhares de variáveis para fazer nossas previsões.

### 10.3) Descrição

- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
- Lecture 4.1 — Linear Regression With Multiple Variables - (Multiple Features) - 8:23
- Lecture 4.2 — Linear Regression With Multiple Variables -- (Gradient Descent For Multiple Variables) - 5:05
- Lecture 4.3 — Linear Regression With Multiple Variables | Gradient In PracticeaI Feature Scaling - 8:53
- Lecture 4.4 — Linear Regression With Multiple Variables | Gradient In PracticeaI | Learning Rate - 8:59
- Lecture 4.5 — Linear Regression With Multiple Variables | Features And Polynomial Regression - 7:40
- Lecture 4.6 — Linear Regression With Multiple Variables | Normal Equation - 16:18
- Lecture 4.7 — Linear Regression With Multiple Variables | Normal Equation Non Invertibility - 5:59

## 11) Implementing from ground up: Regressão Linear Multivariada + POO

### 11.1) Requisitos sugeridos

Atividades 8 e 10.

### 11.2) Motivação

A ideia aqui é ficar um pouco mais familiar com a forma como o scikit-learn funciona, treinando POO.

### 11.3) Descrição

A ideia dessa atividade é estruturar de forma mais elegante o que você fez na atividade 8, colocando dentro de uma classe, ao mesmo tempo que você tenta generalizar a regressão linear para uma dimensão qualquer.

## 12) Introdução ao Pandas

### 12.1) Requisitos sugeridos

Atividade 5.

### 12.2) Motivação

Pandas é a biblioteca de manipulações mais utilizada para estruturar seus dados em Python. Aliada ao Spark e ao SQL, você terá um stack muito robusto para as diferentes tarefas e cenários de manipulação de dados. Masterizar o Pandas é talvez a mais importante na prática e ajudará a aprender as outras com mais facilidade.

### 12.3) Descrição

- [Pandas for Data Science in 20 Minutes | Python Crash Course](https://www.youtube.com/watch?v=tRKeLrwfUgU) (23 min) 
- [Complete Python Pandas Data Science Tutorial! (Reading CSV/Excel files, Sorting, Filtering, Groupby)](https://youtu.be/vmEHCJofslg)  (~1 hora) 
- [Python pandas — An Opinionated Guide](https://youtube.com/playlist?list=PLgJhDSE2ZLxaENZWWF_VOUa5886KiUd15) (~2 horas)

## 13) Algumas métricas de regressão

### 13.1) Requisitos sugeridos

Atividade 10.

### 13.2) Motivação

pass

### 13.3) Descrição

- [Regression Metrics Review I - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=30)
- [Regression Metrics Review II - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=31)

## 14) Aula de Regressão Logística do curso (básico) do Andrew

### 14.1) Requisitos sugeridos

Atividade 10.

### 14.2) Motivação

pass

### 14.3) Descrição

- [Machine Learning — Andrew Ng, Stanford University [FULL COURSE]](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
- Lecture 6.1 — Logistic Regression | Classification - 8:09
- Lecture 6.2 — Logistic Regression | Hypothesis Representation - 7:25
- Lecture 6.3 — Logistic Regression | Decision Boundary - 14:50
- Lecture 6.4 — Logistic Regression | Cost Function - 11:26
- Lecture 6.5 — Logistic Regression | Simplified Cost Function And Gradient Descent - 10:15
- Lecture 6.6 — Logistic Regression | Advanced Optimization - 14:07
- Lecture 6.7 — Logistic Regression | MultiClass Classification OneVsAll - 6:16

## 15) Algumas métricas de classificação

### 15.1) Requisitos sugeridos

Atividade 14.

### 15.2) Motivação

pass

### 15.3) Descrição

- [The 5 Classification Evaluation metrics every Data Scientist must know](https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226)
- [Classification Metrics Review - How to Win a Data Science Competition Learn from Top Kagglers](https://www.bilibili.com/video/BV117411y7Fa?p=32)

## 16) Capítulo/Aula 2 do livro/curso Introduction to Statistical Learning

### 16.1) Requisitos sugeridos

pass

### 16.2) Motivação

O livro [ISL](https://www.ime.unicamp.br/~dias/Intoduction%20to%20Statistical%20Learning.pdf)...

### 16.3) Descrição

- 