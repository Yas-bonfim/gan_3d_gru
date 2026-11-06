# GAN 3D com GRU - Geração de Dígitos em 3D

Este repositório contém o código do projeto de Inteligência Artificial focado na criação de uma Rede Generativa Adversarial (GAN) híbrida, utilizando unidades recorrentes (GRU) e convoluções 3D (Conv3D) para gerar modelos tridimensionais a partir do dataset 3D MNIST.

## Autores


Professor: Flávio Conterato

Yasmin da Silva Bonfim

Gabriel Sete Ribeiro Lago dos Santos

Gustavo Oliveira Ramos Cruz


 ## Sobre o Projeto e Arquitetura

O objetivo principal deste projeto é investigar a capacidade de uma arquitetura GAN gerar nuvens de pontos / voxels que representam dígitos numéricos (0 a 9) em três dimensões.

O modelo é composto por duas redes principais que competem entre si:

Gerador (Generator):

Recebe um vetor de ruído aleatório (distribuição normal).

Utiliza camadas GRU (Gated Recurrent Unit) para processamento inicial do ruído.

Transforma a saída recorrente num tensor 3D e aplica camadas de convolução transposta (Conv3DTranspose) e UpSampling3D para aumentar a resolução até atingir o formato final de 16x16x16x1 (representação do Voxel).

Ativação: LeakyReLU.

Discriminador (Discriminator):

Recebe imagens 3D reais (do dataset) e imagens falsas (do Gerador).

Utiliza camadas Conv3D e MaxPooling3D para extrair características espaciais tridimensionais.

Ativação final com sigmoid para classificar a entrada como Real (1) ou Falsa (0).

## Conjunto de Dados (Dataset)

O projeto utiliza o 3D MNIST, disponibilizado no Kaggle (daavoo/3d-mnist). Trata-se de uma representação tridimensional do clássico dataset MNIST, onde os dígitos são formatados como nuvens de pontos (.h5) e convertidos para uma representação de Voxels (16x16x16) durante a fase de pré-processamento.

## Bibliotecas Utilizadas

Linguagem: Python

Deep Learning: TensorFlow e Keras (Model, Sequential, Conv3D, GRU, Conv3DTranspose)

Processamento de Dados: NumPy, Pandas, h5py

Visualização: Matplotlib (Voxels), Plotly (Gráficos interativos de dispersão 3D)

Outras Ferramentas: Kaggle API, tqdm (barras de progresso)

## Como Executar

O código foi desenvolvido nativamente para rodar no Google Colab.

É necessário ter um token de API do Kaggle (kaggle.json) para realizar o download automático do dataset.

Importe o notebook para o Colab ou clone este repositório.

Ao rodar o código, coloque o seu arquivo kaggle.json no ambiente quando solicitado ou certifique-se de que ele está configurado na pasta ~/.kaggle/.

O script irá extrair os dados .h5, reformatá-los para tensores 3D e iniciar a função de treino training(epochs=100, batch_size=100).

## Resultados

Durante o treino, o modelo gera os seguintes artefactos:

Gráfico de Loss comparativo entre o Discriminador e o Gerador por época.

Projeções visuais em 3D dos Voxels gerados utilizando o matplotlib.

Os pesos e arquitetura do modelo treinado são exportados nos ficheiros gan_gru.h5 e gan_gru.csv.
