# Learning log

## Fase 1 — Definição do problema

**O que aprendi**
- Unidade de observação = uma linha = um vídeo (título + descrição).
- MVP é multiclass (6 classes), não “sensitive sim/não”.
- Se houver violência, a label é Sensitive (prioridade de label).
- Accuracy mente quando a classe perigosa é rara: o baseline sobe; o recall
  de Sensitive do bobinho fica zero.
- Maximizar só recall também quebra: um modelo que sempre grita Sensitive
  bloqueia o inventário inteiro.

**Conceitos novos**
- Multiclass vs multilabel vs binary
- Labeling policy / prioridade de label
- Data leakage
- Class imbalance
- Macro-F1 como guarda-corpo
- Label noise (guerra no jornal)

**Erros que cometi**
- Tratei o YouTube multilabel como se o MVP tivesse que ser multilabel.
- Disse que o baseline de accuracy ficaria baixo; na verdade ele sobe.
- Descrevi o modelo como binário e como multiclass ao mesmo tempo.
- Inverti a prioridade (política na frente de Sensitive).

**Como corrigi**
- Separei fenômeno (vídeo pode ter 2 assuntos) de escolha do MVP (1 label).
- Recalculei o majority baseline: mais raro o Sensitive, mais alta a accuracy.
- FIFA violento → Sensitive; jornal político sem violência → News/Politics.

**O que ainda preciso revisar**
- Precision vs recall com um exemplo numérico meu.
- Por que categoria do YouTube em X é leakage.
- Macro-F1 vs accuracy, em uma frase de entrevista.

**Três perguntas de revisão**
1. Qual é o y de um gameplay violento?
2. Por que accuracy do majority baseline sobe se Sensitive fica mais raro?
3. O que acontece se otimizarmos só recall de Sensitive?

## Fase 2 — Um dataset (YouTube Trending) e suitability

**O que aprendi**
- Duas tabelas sem chave em comum não são um produto; são dois treinos.
- O modelo não junta linhas: ele aprende *padrões* e aplica num texto *novo*.
- YouTube Trending não tem label de violência. Safety com essa tabela seria mentira.
- O que dá para vender: categoria + regra da marca (suitability / targeting).
- Nike/banco/cereal não estão no CSV. O modelo prevê categoria; o sim/não é policy.

**Conceitos novos**
- Brand safety vs brand suitability
- Training vs serving
- Policy / allow list depois do modelo
- Domain shift (por que Jigsaw + YouTube não se “ensinam”)

**Erros que cometi**
- Quis ligar comentários da Wikipedia aos vídeos com um JOIN que não existe.
- Achei que o Modelo A ensinava o B.
- Achei que o B classificaria unsafe, mesmo com trending já filtrado.

**Como corrigi**
- Um dataset, um grain, um modelo de categoria.
- Match da marca = lista permitida, não segundo `.fit`.

**O que ainda preciso revisar**
- Por que `category_id` no X é leakage.
- Macro-F1 vs accuracy no trending (Entertainment costuma dominar).

**Três perguntas de revisão**
1. O modelo prevê “Nike sim/não” ou a categoria?
2. Se o cereal aceita Entertainment e o modelo chama News de Entertainment, o que acontece?
3. Views e nome do canal devem entrar em X? Por quê?

## Fase 3 — Split, TF-IDF, precision vs recall

**O que aprendi**
- Train e test não se JOIN-am. A ponte é o `vec` e o `model` treinados no caderno e usados na prova.
- Precision e recall são os dois no **test**. Muda o denominador, não o split.
- Recall News = dos que ERAM News, quantos peguei. Precision = dos que DISSE News, quantos eram.
- Accuracy 0.80 com bobinho 0.52 parece ótimo; macro-F1 0.67 mostra Gaming/Education fracos.
- `class_weight="balanced"`: News recall 0.71→0.89, Gaming 0.14→0.48, accuracy quase igual, precision de News cai.

**Conceitos novos**
- `train_test_split` + `stratify`
- TF-IDF (`fit` só no treino)
- Logistic Regression (`fit` / `predict`)
- macro-F1 vs accuracy vs weighted avg
- `class_weight="balanced"`

**Erros que cometi**
- Achei que precision era test e recall era train.
- Achei que os dois % de News eram a mesma conta.

**Como corrigi**
- Mesma prova, duas pilhas: 104 gabaritos vs ~83 chutes.
- Compare sempre o bobinho no **test**, não nos 6351.

**O que ainda preciso revisar**
- Célula da matriz: News previsto como Entertainment.
- Por que 21 Gaming na prova deixa o recall instável.

**Três perguntas de revisão**
1. Onde train e test se conectam, se as linhas não se juntam?
2. Denominador do recall de News: 104 ou os chutes?
3. Por que aceitamos accuracy 0.79 com `balanced` em vez de 0.80 sem peso?

## Fase 4 — Error analysis, n-grams, threshold

**O que aprendi**
- Error analysis = olhar o vídeo, não só o relatório. O cereal se importa com News → Entertainment.
- n-grams `(1, 2)`: macro-F1 quase igual (0.718 → 0.716); false yes do cereal 4 → 2. O produto mudou; a métrica global não.
- Os 2 que sobraram: Infowars (formato de talk show, erro do modelo) e Rose/Time's Up (label duvidoso).
- `predict` = campeão. `predict_proba` = 6 notas. Policy do cereal: `p_allow = P(Sports)+P(Entertainment)`.
- Milo `p_allow ≈ 0.40`, Rose ≈ 0.47. Corte 0.55 zerou News no jornal e bloqueou 254 Entertainment (481 SIM / 1271).
- O threshold é da marca, não do sklearn. Criança → mais alto; Nike → 0.50 basta.

**Conceitos novos**
- Error analysis / false yes de produto
- n-grams
- `predict_proba` vs `predict` (argmax)
- Threshold / confidence policy
- Tradeoff leak vs inventário

**Erros que cometi**
- Li “não mudou nada” no macro-F1 e ignorei o 4 → 2.
- Hipótese Late Night: os 4 primeiros erros não eram talk show; o Infowars depois sim.

**Como corrigi**
- Métrica do cereal = count News publicado, não accuracy.
- Policy em cima do modelo, em vez de outro `.fit`.

**O que ainda preciso revisar**
- Por que 6 notas somam 1 (softmax da logística).
- Por que `category_id` no X continua leakage.

**Três perguntas de revisão**
1. Se macro-F1 fica igual e o leak do cereal cai, o experimento valeu?
2. O que `predict` devolve no Milo vs o que `p_allow` devolve?
3. Por que o cereal usa 0.55 e a Nike pode usar 0.50?
