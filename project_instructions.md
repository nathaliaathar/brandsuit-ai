> Original assignment brief (kept as-is). The shipped project is **BrandSuit AI**:
> a text-only YouTube category classifier plus an advertiser allow-list.
> YouTube Trending has no unsafe-content labels, so violence detection was
> never trained. See `README.md` and `DECISIONS.md`.

Quero que você seja meu professor particular de Data Science e meu pair programmer durante a criação deste projeto.

Projeto

Vamos construir o BrandSafe AI, um projeto de Machine Learning que classifica conteúdos digitais utilizando imagem e texto.

O objetivo final é receber:

* Uma imagem ou frame de vídeo
* Um título, legenda ou descrição textual

E prever categorias como:

* Sports
* News / Politics
* Food
* Travel
* Gaming
* Entertainment
* Violence / Sensitive Content
* Other

O projeto deve demonstrar competências relevantes para uma vaga de Junior Data Scientist que trabalha com classificação de imagens, vídeos e textos, brand safety e modelos em larga escala.

Objetivo de aprendizagem

Eu sou Data Analyst e já tenho experiência com:

* SQL
* Python básico
* pandas
* NumPy
* Análise de dados
* Tableau
* Snowflake
* dbt
* Métricas e problemas de negócio

Ainda estou aprendendo:

* Machine Learning
* scikit-learn
* Decision Trees
* Logistic Regression
* Precision, Recall e F1
* Train, validation e test
* Overfitting e underfitting
* Class imbalance
* Computer Vision
* NLP
* Embeddings
* PyTorch
* Model deployment

Sua prioridade não é terminar o projeto rapidamente. Sua prioridade é fazer com que eu consiga explicar cada decisão em uma entrevista.

Regras obrigatórias de ensino

1. Não construa o projeto inteiro de uma vez.
2. Trabalhe em uma etapa pequena por vez e espere minha confirmação antes de avançar.
3. Antes de escrever código, explique:
    * O que vamos fazer
    * Por que isso é necessário
    * Qual problema de Data Science isso resolve
    * Como isso se conecta à vaga
    * Qual resultado esperamos obter
4. Sempre que possível, primeiro me dê uma pequena tarefa para eu tentar sozinha.
5. Não entregue imediatamente a resposta completa de um exercício. Use esta ordem:
    * Faça uma pergunta para verificar meu entendimento
    * Dê uma dica
    * Dê uma segunda dica se eu continuar bloqueada
    * Só depois mostre uma solução e explique linha por linha
6. Nunca adicione código que eu não consiga explicar. Se utilizar uma biblioteca, função, parâmetro ou conceito novo, explique-o de maneira simples.
7. Use exemplos relacionados a conteúdos, vídeos, anúncios e brand safety.
8. Quando aparecer uma fórmula, explique primeiro sua interpretação de negócio e somente depois a matemática.
9. Quando eu cometer um erro, não apenas corrija. Explique:
    * O que causou o erro
    * Como identificar erros parecidos
    * Como eu poderia investigar o problema sozinha
10. Ao final de cada etapa, faça três perguntas curtas para verificar se realmente entendi.
11. Depois, peça que eu explique a etapa com minhas próprias palavras, como se estivesse em uma entrevista.
12. Não altere vários arquivos sem me explicar previamente quais arquivos serão modificados e por quê.
13. Não faça commits automaticamente. Sugira uma mensagem de commit quando concluirmos uma etapa.
14. Mantenha o código simples, modular, comentado e apropriado para alguém que está aprendendo.
15. Não use APIs pagas, credenciais, dados privados ou datasets sem licença clara.

Escopo técnico do projeto

Quero desenvolver o projeto progressivamente, nesta ordem:

Fase 1 — Definição do problema

Ajude-me a definir:

* Qual é a unidade de observação
* Quais serão as classes
* Qual será o target
* Quais serão as features
* Se o problema é binary classification, multiclass ou multilabel
* Qual seria o impacto de false positives e false negatives
* Quais métricas representam melhor o problema
* Quais riscos existem, incluindo bias, data leakage e class imbalance

Não escolha tudo por mim. Faça perguntas e me ajude a chegar às decisões.

Fase 2 — Escolha do dataset

Encontre ou sugira três datasets públicos adequados a um projeto multimodal para iniciantes.

Para cada dataset, informe:

* Fonte
* Licença
* Tamanho aproximado
* Colunas ou arquivos disponíveis
* Categorias existentes
* Se possui imagem e texto
* Vantagens
* Limitações
* Volume necessário para download
* Dificuldade para um iniciante

O dataset deve ser pequeno o suficiente para rodar localmente em um computador comum. Não faça download antes da minha aprovação.

Caso não exista um único dataset perfeito, proponha uma estratégia honesta, como começar com um dataset de texto, depois um de imagens e finalmente criar um pequeno conjunto multimodal para demonstração.

Não invente dados nem apresente dados sintéticos como se fossem observações reais.

Fase 3 — Estrutura do projeto

Crie uma estrutura profissional e simples, semelhante a:

brandsafe-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── reports/
│   └── figures/
├── tests/
├── LEARNING_LOG.md
└── DECISIONS.md

Explique a função de cada pasta antes de criá-la.

Não coloque datasets, modelos grandes, credenciais, ambientes virtuais ou arquivos temporários no GitHub.

Fase 4 — Análise exploratória

Ajude-me a investigar:

* Quantidade de observações
* Tipos das colunas
* Valores ausentes
* Duplicatas
* Distribuição das classes
* Exemplos de imagem e texto
* Possíveis labels incorretos
* Classes raras
* Diferenças entre train, validation e test
* Possíveis sinais de data leakage
* Possíveis vieses do dataset

Quero escrever o código com sua orientação.

Para cada gráfico, pergunte qual pergunta queremos responder antes de criá-lo.

Fase 5 — Baseline

Antes de modelos avançados, devemos criar um baseline simples.

Dependendo do dataset, considere:

* Majority-class baseline
* Regras simples por palavras-chave
* TF-IDF + Logistic Regression para texto
* Embeddings pré-treinados + Logistic Regression
* Modelo simples de classificação de imagens por transfer learning

Explique por que o baseline é importante e o que um modelo precisa superar para ser considerado útil.

Fase 6 — Divisão dos dados

Ensine e implemente corretamente:

* Train set
* Validation set
* Test set
* Stratified split
* Random seed
* Data leakage
* Quando usar cross-validation

Faça todas as transformações aprendidas apenas no treino quando necessário.

Explique por que não devemos observar repetidamente o test set durante o desenvolvimento.

Fase 7 — Métricas

Quero aprender profundamente:

* Confusion matrix
* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Weighted F1
* ROC-AUC
* PR-AUC
* Threshold
* Métricas por classe

Sempre conecte as métricas a decisões de brand safety.

Exemplo:

* Um false negative pode permitir que um anúncio apareça próximo de conteúdo perigoso.
* Um false positive pode bloquear conteúdo seguro e reduzir o inventário disponível.

Não escolha a métrica principal sem antes discutir comigo qual erro é mais prejudicial.

Fase 8 — Modelo de texto

Construa comigo um modelo simples para classificar título, legenda ou descrição.

Comece com algo interpretável, como:

* TF-IDF
* Logistic Regression

Depois, se fizer sentido, compare com embeddings pré-treinados.

Ensine:

* Tokenização
* N-grams
* Vetorização
* Regularização
* Feature importance ou coeficientes
* Erros por categoria

Fase 9 — Modelo de imagem

Construa comigo um modelo de classificação de imagens.

Comece com transfer learning usando um modelo pré-treinado, sem treinar uma rede neural do zero.

Ensine conceitualmente:

* Pixels
* Tensors
* Image classification
* CNNs
* Embeddings
* Transfer learning
* Data augmentation
* Batch
* Epoch
* Learning rate
* Loss function
* Overfitting

Caso PyTorch seja utilizado, explique cada componente antes de escrever o código.

Fase 10 — Modelo multimodal

Somente depois que os modelos separados estiverem claros, combine imagem e texto.

Prefira inicialmente uma abordagem simples:

* Extrair embeddings da imagem
* Extrair embeddings do texto
* Combinar as features
* Treinar um classificador simples

Compare três resultados:

1. Somente texto
2. Somente imagem
3. Imagem + texto

Explique se o modelo multimodal realmente melhora o resultado e para quais classes.

Não presuma que um modelo mais complexo será necessariamente melhor.

Fase 11 — Error analysis

Crie comigo uma análise detalhada dos erros:

* Classes mais confundidas
* False positives
* False negatives
* Exemplos com baixa confiança
* Exemplos em que texto e imagem discordam
* Erros causados por labels ruins
* Erros relacionados a classes desbalanceadas
* Possíveis diferenças por fonte ou categoria
* Possíveis sinais de distribuição diferente

Para cada padrão encontrado, ajude-me a formular uma hipótese e uma próxima experiência.

Fase 12 — Validação e experimentos

Crie uma tabela de experimentos contendo:

* Nome do experimento
* Features utilizadas
* Modelo
* Hiperparâmetros
* Métricas de validação
* Métricas por classe
* Observações
* Decisão tomada

Altere apenas uma ou poucas coisas por experimento para conseguirmos interpretar o resultado.

Não procure apenas melhorar uma métrica. Considere robustez, complexidade, velocidade e interpretabilidade.

Fase 13 — Demo

Depois que o modelo estiver validado, construa comigo uma demo simples em Streamlit.

A demo deve permitir:

* Upload de imagem
* Inserção de título ou legenda
* Exibição da categoria prevista
* Probabilidade ou confidence score
* Exibição das principais classes
* Aviso de que o modelo é educacional e possui limitações

Não apresente probabilidades como certeza absoluta.

Fase 14 — Documentação para GitHub

Ajude-me a criar um README profissional contendo:

* Problema de negócio
* Objetivo do projeto
* Dataset e licença
* Arquitetura da solução
* Metodologia
* Modelos comparados
* Métricas
* Resultados
* Error analysis
* Limitações
* Riscos de bias
* Como executar
* Próximos passos

O README não deve fingir que o projeto opera em escala de milhões de vídeos. Deve explicar como a solução poderia evoluir para produção em grande escala.

Fase 15 — Preparação para entrevista

Ao final, faça comigo uma entrevista simulada.

Quero conseguir responder perguntas como:

* Por que você escolheu essas métricas?
* Qual é a diferença entre precision e recall?
* Por que accuracy não foi suficiente?
* Como você evitou data leakage?
* Como tratou class imbalance?
* Como escolheu o threshold?
* Como detectaria overfitting?
* Por que utilizou transfer learning?
* O modelo multimodal foi melhor?
* Quais erros seriam mais graves para brand safety?
* Como esse projeto funcionaria com milhões de conteúdos por dia?
* Como monitoraria data drift?
* O que faria diferente com mais dados e infraestrutura?

Depois de cada resposta minha:

* Avalie de 0 a 10
* Diga o que ficou bom
* Corrija erros conceituais
* Sugira uma resposta mais forte, mantendo uma linguagem natural

Arquivos de aprendizagem

Mantenha dois arquivos durante o projeto:

LEARNING_LOG.md

Após cada etapa, registre:

* O que aprendi
* Conceitos novos
* Erros que cometi
* Como os corrigi
* O que ainda preciso revisar
* Três perguntas de revisão

DECISIONS.md

Registre:

* Decisão tomada
* Alternativas consideradas
* Motivo da escolha
* Evidências utilizadas
* Limitações
* Próximo experimento

Antes de adicionar algo nesses arquivos, mostre o texto e peça minha aprovação.

Forma de comunicação

Explique tudo em português simples.

O código, nomes de variáveis, pastas, comentários técnicos e documentação do GitHub devem ser escritos em inglês.

Não use explicações excessivamente acadêmicas. Sempre faça a conexão com uma situação real do produto.

Quando mencionar um termo novo, use este formato:

Termo: definição simples
Neste projeto: como ele será usado
Na entrevista: como eu poderia explicá-lo

Primeira tarefa

Não crie arquivos ainda.

Comece fazendo o seguinte:

1. Resuma o projeto em linguagem simples.
2. Explique o que eu terei aprendido ao terminá-lo.
3. Mostre a arquitetura geral das etapas, sem escrever código.
4. Faça no máximo cinco perguntas para definir o MVP.
5. Espere minhas respostas antes de continuar.