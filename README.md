# Classificador_de_Imagem

Trabalho de estudo dirigido sobre redes neurais. Implementação de um programa para reconhecimento de imagens por meio de uma CNN(rede neural convolucional), com uma interface em javascript para que o usuário possa testar o funcionamento do treinamento.

# Base de dados

É importante lembrar que os arquivos usados para o treinamento e teste da rede neural são muito grandes, portanto, para o funcionamento do programa é necessário realizar o download dos arquivos "banana.npy", "cake.npy", "grapes.npy", "hamburger.npy", "pineapple.npy" e "pizza.npy", da database do jogo "Quick, Draw!" na nuvem da Google, acessível pelo link: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=true

# Bibliotecas

Para a execução deste projeto, são necessárias as bibliotecas:
- numpy;
- pandas;
- pickle;
- seaborn;
- matplotlib;
- sklearn;
- keras;
- flask;

# Execução

Para executar o treinamento da CNN, basta entrar na pasta do projeto e executar o arquivo *CNN_Training.py* com o python;
Para verificar a porcentagem de acertos do último treinamento realizado, executar o arquivo *CNN_Evaluate.py*;
Para executar a interface para desenho e consequentemente, teste, do programa, executar o arquivo *app.py* e abrir o navegador no endereço fornecido no terminal.

Desenvolvedores: Eryk Kooshin Suguiura & Vitor José Duarte Quintans.
