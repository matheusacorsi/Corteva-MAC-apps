# EPG Classifier — Diaphorina citri / psilídeos

App em Streamlit para classificação automática de waveforms de EPG
(Electrical Penetration Graph) a partir de arquivos brutos `.D01`, `.D02`, ...
(cabeçalho ASCII + corpo de floats32 little-endian, como gerado por sistemas
DC-EPG estilo Wageningen/Giga-8).

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`.

## Como usar

1. Envie na barra lateral um ou mais arquivos `.DXX`. Arquivos com o mesmo
   nome-base (tudo antes de `.D01`, `.D02`, ...) são reconhecidos como
   segmentos consecutivos do **mesmo canal/inseto** e concatenados
   automaticamente na ordem correta.
2. Ajuste (se necessário) o tamanho de janela, sobreposição e os thresholds
   de classificação na barra lateral — os valores padrão seguem
   Bonani et al. (2010).
3. Explore as abas:
   - **Visão geral**: pizza de proporção de tempo por waveform + tabela de
     métricas (NWEI/WDI/WDE) + comparação com médias da literatura.
   - **Linha do tempo (Gantt)**: visualização estilo Gantt dos eventos
     classificados ao longo do tempo.
   - **Traço do sinal**: sinal bruto (tensão média por janela) com
     sombreamento colorido por waveform, e zoom para inspeção detalhada.
   - **Eventos detalhados**: tabela filtrável de todos os eventos (runs)
     classificados.
   - **Exportar**: download em CSV das features, eventos e resumo.

## Modos de extração de features

O app oferece dois métodos, selecionáveis na barra lateral:

### Avançado (estilo DiscoEPG) — recomendado

Implementado em `epg_features.py`. Para cada janela deslizante, aplica:

1. **Pré-processamento**: remoção de deriva de linha de base (mediana móvel
   ~60s) + **padronização robusta por gravação** `(x − mediana) / IQR`. Isso
   torna as amplitudes comparáveis entre gravações com **ganhos de amplificador
   diferentes** — resolvendo o principal ponto de falha da abordagem por
   thresholds absolutos em Volts.
2. **13 features estatísticas/de complexidade** (conjunto DiscoEPG, Dinh et al.
   2024): média, RMS, desvio-padrão, variância, assimetria, quantis
   0.05/0.25/0.5/0.75/0.95, taxa de cruzamento por zero, entropia de Shannon e
   entropia de permutação.
3. **Features espectrais**: frequência dominante, centroide espectral, entropia
   espectral e razões de potência nas bandas 1–4 / 5–9 / 11–19 Hz.
4. **Features de wavelet (DWT db4)**: energia relativa e entropia por nível de
   decomposição — capturam a estrutura transitória de transições E1→E2 e pd's
   que a FFT janelada borra.

Total: **26 features por janela**. No modo avançado, os thresholds do
classificador heurístico passam a operar em **unidades robustas (múltiplos de
IQR)**, não em Volts — por isso o app usa um conjunto de thresholds distinto
(`DEFAULT_THRESHOLDS_STD`).

A aba **🔬 Explorador de features** mostra boxplots de cada feature por waveform
e a matriz de correlação — útil tanto para calibrar thresholds quanto para
selecionar features para um modelo de ML.

### FFT + amplitude (legado)

Modo original: apenas frequência dominante (FFT) e amplitude pico-a-pico
absoluta em Volts. Mantido para comparação A/B.

## Treinando um classificador supervisionado (ML) com anotações .ANA do Stylet+

Se você já tem gravações anotadas visualmente no Stylet+/Stylet+a (arquivos
`.ANA`), pode treinar um classificador supervisionado que tende a superar a
heurística de thresholds — este é o caminho recomendado para produção.

### Formato do arquivo `.ANA`

Texto tab-delimitado, codificado em **UTF-16LE** (com BOM), 3 colunas por
linha: **código do waveform**, **tempo de início (s)**, **voltagem de saída
(mV)**. Cada linha é um evento (clique manual no Stylet+): o código vale do
seu tempo de início até o início do próximo evento — ou seja, **é um rótulo
por evento, não por amostra**, e `epg_ana.py` expande isso em um rótulo
contínuo alinhado com as janelas de features.

**Importante:** os códigos numéricos dos botões são configuráveis por
experimento no Stylet+ (não há um mapeamento universal). Ajuste
`DEFAULT_CODE_MAP` em `epg_ana.py` para bater com a configuração de botões
usada nas suas anotações, ou passe seu próprio `code_map` ao chamar as
funções.

### Estrutura de pastas esperada

Coloque em uma pasta os arquivos `.DXX` e o `.ANA` correspondente, com o
**mesmo nome-base**:

```
dataset/
  PsilideoMudaT1-3-ch6.D01
  PsilideoMudaT1-3-ch6.D02
  ...
  PsilideoMudaT1-3-ch6.D09
  PsilideoMudaT1-3-ch6.ANA   <- mesma anotação p/ todos os segmentos do canal
  PsilideoMudaT1-4-ch2.D01
  ...
  PsilideoMudaT1-4-ch2.ANA
```

### Rodando o treino

```bash
python train_model.py --data-dir dataset/ --out modelo_epg.joblib
```

O script (`train_model.py`):

1. Agrupa os `.DXX` por canal/gravação e localiza o `.ANA` correspondente.
2. Extrai as 26 features avançadas (`epg_features.extract_features_advanced`)
   para cada gravação.
3. Expande as anotações de evento em rótulo contínuo e rotula cada janela de
   features (`epg_ana.build_labeled_dataset`).
4. Faz o split treino/teste **por gravação** (`GroupShuffleSplit`), nunca por
   janela isolada — janelas do mesmo inseto são autocorrelacionadas, e um
   split aleatório por janela vazaria informação e infla artificialmente a
   acurácia.
5. Treina um **XGBoost** (100 árvores, learning rate 0.3, profundidade 6 —
   configuração usada por Willett et al. 2016 para *D. citri*), com pesos de
   classe para compensar o desbalanceamento natural (Np/C dominantes, E1/D
   raros).
6. Reporta `classification_report`, matriz de confusão e F1-macro no conjunto
   de teste (gravações nunca vistas no treino).
7. Salva o modelo, o label encoder e as colunas de feature em um único
   `.joblib`, pronto para reuso.

### Resultado real (1 inseto anotado, D. citri, canal 6, 8,1h)

Treinamos com a anotação real deste dataset (mapeamento de código confirmado:
1=Np, 2=C, 3=D, 4=E1, 5=E2, 7=G — código 6 não usado, código 99 é marcador de
fim de arquivo e é automaticamente excluído). Distribuição real de classes:
Np 61,7% · E2 20,6% · C 10,3% · G 5,2% · D 1,2% · E1 1,0%.

Como só havia 1 gravação anotada, `GroupShuffleSplit` (split por inseto) não é
aplicável — o script detecta isso automaticamente e usa um **split por blocos
temporais alternados** (distribuídos por toda a gravação) como aproximação,
emitindo um aviso explícito de que isso não valida generalização entre
insetos diferentes.

| Waveform | F1-score | Observação |
|---|---|---|
| Np | 0,98 | Excelente |
| E2 | 0,98 | Excelente |
| C  | 0,93 | Muito bom |
| G  | 0,93 | Muito bom — a heurística de thresholds nunca detectava G neste canal |
| E1 | 0,70 | Razoável, poucas amostras (20 no teste) |
| D  | 0,20 | Fraco — confundido com C, E2 e Np |

F1-macro: 0,786. O desempenho fraco em D é esperado e biologicamente
coerente: é a waveform mais breve (~46s de média na literatura), e a
confusão com waveforms adjacentes é o mesmo padrão relatado pelo DiscoEPG
para waveforms transitórias curtas (ex: pd em afídeos).

**Recomendação para produção:** anote mais insetos (idealmente de diferentes
sessões/dias) antes de considerar este modelo validado — um modelo treinado
em 1 único inseto captura tanto o padrão biológico quanto idiossincrasias
daquele indivíduo específico (posição do eletrodo, ganho daquela sessão),
e pode não generalizar.

### Usando o modelo treinado no app Streamlit

O `app.py` agora aceita o upload de um modelo `.joblib` (gerado por
`train_model.py`) diretamente na barra lateral, em **🤖 Modelo treinado
(opcional)**. Quando um modelo é carregado:
- A classificação heurística (thresholds) é ignorada — os waveforms exibidos
  em todas as abas vêm das predições do modelo.
- É necessário estar no modo **Avançado (estilo DiscoEPG)**, com o mesmo
  tamanho de janela/sobreposição usados no treino, para que as 26 features
  batam com o que o modelo espera.

## Metodologia e limitações

A classificação é feita por uma **heurística automática de primeira
passagem**: para cada janela deslizante do sinal, calcula-se a frequência
dominante (via FFT, restrita a 0.5–25 Hz) e a amplitude pico-a-pico
(percentil 95 − percentil 5), e compara-se com as faixas descritas para
cada waveform em:

> Bonani, J.P., Fereres, A., Garzo, E., Miranda, M.P., Appezzato-Da-Gloria, B.,
> Lopes, J.R.S. (2010). Characterization of electrical penetration graphs of
> the Asian citrus psyllid, *Diaphorina citri*, in sweet orange seedlings.
> *Entomologia Experimentalis et Applicata*, 134: 35–49.

**Isso NÃO substitui a anotação visual especializada** feita em softwares
dedicados (Stylet+a, PROBE 3.5) por um analista treinado — especialmente
porque:

- A amplitude relativa depende do ganho do amplificador (desconhecido a
  priori); os thresholds em Volts podem precisar de calibração.
- Frequência dominante por FFT em janelas curtas é sensível a ruído
  elétrico e pode confundir waveforms com faixas de frequência próximas.
- A heurística não considera a forma de onda (formato dos picos), apenas
  frequência/amplitude agregadas — os artigos originais também usam
  critérios visuais.

Use os resultados como **triagem exploratória** e ponto de partida — ideal
para comparar tratamentos rapidamente ou para gerar hipóteses — mas valide
com inspeção visual antes de usar os números em publicações.

## Extensões futuras sugeridas

- Treinar um classificador supervisionado (ex: Random Forest, 1D-CNN) usando
  trechos anotados manualmente como ground truth.
- Adicionar suporte a múltiplos canais lado a lado para comparação de
  tratamentos (T1, T2, T3...) em um único relatório.
- Exportar relatório em PDF com os gráficos principais.
