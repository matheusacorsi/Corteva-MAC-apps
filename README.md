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
