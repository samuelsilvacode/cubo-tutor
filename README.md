# cubo-tutor

Mostre o cubo mágico para a webcam, o programa lê as cores e te ensina a montar, giro por giro, pelo método de camadas.

Enquanto você mexe no cubo, a câmera confere a face da frente e avança sozinha quando o giro foi feito.

## Como funciona

1. **Escanear**: mostre as 6 faces para a câmera, uma por vez (amarelo em cima).
2. **Conferir**: o programa monta a planificação e corrige sozinho leituras impossíveis (escolhe o cubo válido mais parecido com o que a câmera viu). A aula começa sozinha; se quiser, clique numa cor para trocar.
3. **Aula**: 8 etapas do método de camadas, cada uma com o objetivo, a dica do passo atual, o giro e o porquê dele funcionar.

Também mostra a solução mais curta (algoritmo de duas fases de Kociemba, ~20 giros) como curiosidade.

## Rodando

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Sem câmera, para testar: `python main.py --demo`

Na primeira vez que a solução curta é calculada, o pacote gera umas tabelas (~70 MB, uns 20 a 30 minutos). Depois fica instantâneo.

## Atalhos

| Tela | Tecla | Ação |
|---|---|---|
| Escanear | (nada) | segure a face parada ~1 s, captura sozinho |
| | ESPAÇO | captura na hora |
| | BACKSPACE | refaz a anterior |
| Conferir | clique | troca a cor do quadrado |
| | ENTER | começa a aula |
| Aula | → / ESPAÇO, ← | próximo / anterior |
| | N / P | próxima / anterior etapa |
| | A | liga/desliga o avanço automático |
| | O | mostra a solução curta |
| Todas | M | espelha a câmera |
| | Q | sai |

## Detalhes técnicos

- O cubo é modelado por adesivos com posição e normal 3D; cada giro é uma rotação de vetores.
- As cores são classificadas usando os próprios centros do cubo como referência, com a restrição de 9 adesivos por cor (problema de atribuição resolvido com `scipy`).
- Antes da aula, o estado é validado: cores repetidas, peças impossíveis, cantos torcidos, meios virados e paridade.
- `pytest` resolve 300 embaralhamentos aleatórios para garantir que o solver sempre termina.

### Vermelho, laranja e amarelo

São as cores que mais confundem com webcam e luz amarelada. O que o programa faz:

1. **Balanço de branco** pelo centro branco do cubo, para tirar o tom da lâmpada.
2. Compara as cores pelo **tom** (ângulo de matiz no espaço LAB), dando pouco peso ao brilho, que muda de uma face pra outra.
3. Usa os **centros do próprio cubo** como referência, então funciona com cubos de tons diferentes.
4. A correção pelas peças desfaz o que ainda sobrar de erro.

Para medir: `python tools/estresse_cores.py 300` simula um cubo com vermelho e laranja próximos sob luz quente e exposição variando por face. Antes dessas mudanças: 27% dos cubos lidos 100% certos (72% com correção). Depois: 96% (100% com correção).
