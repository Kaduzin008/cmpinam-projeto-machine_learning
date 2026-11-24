# Classificação de candidatos a pulsares (HTRU2)

Projeto da disciplina CMPINAM — Introdução ao Aprendizado de Máquina.

**Integrantes:**
- Carlos Eduardo Ruzene Nascimento (CP3031349)
- Luis Gabriel Milani da Silva (CP3029514)
- Isac Caria Leite (CP3031306)

## Problema
Classificação binária: dado um candidato captado por radiotelescópio, dizer se é um **pulsar real** (`is_pulsar = 1`) ou **interferência/ruído** (`0`), a partir de 8 estatísticas resumidas do perfil integrado do pulso e da curva DM–SNR. O problema é fortemente desbalanceado (≈ 9,2% de pulsares).

## Base de dados
Dataset 07 — HTRU2, fornecido pelo professor: `data/train.csv` (14.318 candidatos) e `data/test.csv` (3.580). A divisão estratificada fornecida foi preservada.
Fonte: UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/372/htru2 (licença CC BY 4.0). Referência: Lyon, R. (2015). *HTRU2*. https://doi.org/10.24432/C5DK6R

## Metodologia
- **Preparação:** checagem de tipos, duplicatas, valores inválidos e faltantes (nenhum encontrado). Outliers **mantidos**, pois carregam informação da classe (ex.: ≈ 92% dos outliers de `mean_integrated_profile` são pulsares).
- **EDA:** 8 perguntas, 5 tipos de gráfico (barras, histograma, boxplot, mapa de calor, dispersão), usando só o treino.
- **Modelagem:** Baseline (Dummy), Regressão Logística, KNN, SVM (RBF), Random Forest e Gradient Boosting.
- **Sem vazamento:** padronização dentro de `Pipeline`; validação cruzada estratificada (5 dobras); `GridSearchCV` só no treino; o teste é usado uma única vez.
- **Desbalanceamento:** `class_weight` (sem reamostragem). Busca de hiperparâmetros por PR-AUC.
- **Técnica extra:** ajuste do limiar de decisão para maximizar o F1 em predições fora da amostra (`cross_val_predict`) no treino, aplicado ao teste.
- **Métricas:** PR-AUC (principal), precisão, revocação, F1, ROC-AUC e acurácia. `random_state=42`.

## Principais resultados (teste, limiar ajustado no treino)
| Modelo | Precisão | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Gradient Boosting | 0,929 | 0,872 | 0,899 | 0,929 |
| KNN | 0,925 | 0,863 | 0,893 | 0,914 |
| Regressão Logística | 0,895 | 0,887 | 0,891 | 0,932 |
| Random Forest | 0,892 | 0,881 | 0,887 | 0,931 |
| SVM (RBF) | 0,894 | 0,875 | 0,884 | 0,847 |
| Baseline (sempre "ruído") | 0,000 | 0,000 | 0,000 | 0,092 |

O baseline tem acurácia de ≈ 90,8%, mas não encontra nenhum pulsar. Os melhores modelos ficam praticamente empatados (diferenças de poucos casos entre 328 pulsares); a regressão logística, a mais simples, está entre os melhores. Detalhes, interpretação e limitações no notebook.

## Como executar
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook pulsar_htru2.ipynb   # Run All (pode levar alguns minutos)
```
Os caminhos são relativos (`data/...`).
