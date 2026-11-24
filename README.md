# cubo-tutor

Mostre o cubo mágico para a webcam, o programa lê as cores e te ensina a montar, giro por giro, pelo método de camadas.

Enquanto você mexe no cubo, a câmera confere a face da frente e avança sozinha quando o giro foi feito.

## Como funciona

1. **Escanear**: o programa pede uma face por vez pelo centro ("mostre a face com o centro azul"). Pode girar a face como quiser. Ele confere se o centro é mesmo o pedido e avisa se não for; vermelho/laranja e branco/amarelo são comparados entre si, porque a webcam confunde.
2. **Conferir**: o programa monta a planificação e corrige sozinho leituras impossíveis (escolhe o cubo válido mais parecido com o que a câmera viu). A aula começa sozinha; se quiser, clique numa cor para trocar.
3. **Aula**: 8 etapas do método de camadas, cada uma com o objetivo, a dica do passo atual, o giro e o porquê dele funcionar.

**Modo espelho (tecla E):** depois da leitura, uma réplica 3D do cubo segue a posição dele na sua mão em tempo real, inclusive girando 360°. Dois rastreadores se completam:

- **pela grade:** quando uma face aparece de frente, a pose sai da perspectiva dos 9 adesivos (`solvePnP`);
- **pelo modelo:** entre um quadro e outro, a pose anterior é ajustada até as cores projetadas de todos os adesivos visíveis baterem com a imagem. Como o cubo inteiro já foi lido, isso funciona com quina e aresta viradas para a câmera.

**Giro das camadas:** no modo espelho a réplica também gira a camada quando você gira. Quando a nota da pose cai, o programa testa os 18 giros possíveis e aplica o que fizer a imagem bater bem melhor por 3 quadros seguidos (comparando cada candidato já com a pose ajustada para ele). Num teste sintético com 10 giros aleatórios por rodada: 46/50 giros vistos na ordem certa sem dedos na frente, 36/50 com dedos desenhados por cima que não são reconhecidos como mão (`python tools/testa_giros.py`).

A mão na frente do cubo não atrapalha: o esqueleto das mãos (MediaPipe Hands, 21 pontos por mão) aprende a cor da pele a cada quadro e recorta o contorno real dos dedos, que ficam fora da comparação de cores. O modelo da mão (`hand_landmarker.task`, ~8 MB) vai na pasta `modelos/`: baixe de https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task. Sem ele o programa funciona, só não ignora a mão.

Num giro sintético de 360° (3° por quadro) ele rastreia 100% dos quadros, com erro mediano de 2 a 3°, em ~30 ms por quadro.

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
| Escanear | (nada) | mostrar a face pedida |
| | BACKSPACE | refaz a face anterior |
| Conferir | clique | troca a cor do quadrado |
| | ENTER | começa a aula |
| Aula | → / ESPAÇO, ← | próximo / anterior |
| | N / P | próxima / anterior etapa |
| | A | liga/desliga o avanço automático |
| | O | mostra a solução curta |
| Conferir / Aula | E | liga/desliga o modo espelho |
| Todas | M | espelha a câmera |
| | Q | sai |

## Detalhes técnicos

- O detector acha os 9 adesivos em qualquer lugar da imagem (bordas + contornos + encaixe numa grade 3x3), mesmo com o cubo torto. Se não achar, usa um quadrado fixo no meio da tela.
- Cada quadro nítido vira um voto. As leituras são agrupadas por face comparando as 9 cores com o brilho normalizado, então dá pra mostrar as faces em qualquer ordem e repetir.
- No fim, o programa descobre onde fica cada face e como ela estava girada: os centros dizem a posição (branco oposto ao amarelo, verde ao azul, vermelho ao laranja) e, das 4.096 combinações de giro das 6 faces, fica com a que faz as 20 peças existirem.

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

**Calibração do vermelho e do laranja** (`python tools/calibrar.py`): você mostra uma face com vermelho e sem laranja, depois uma com laranja e sem vermelho. Na webcam testada, os dois estouram o canal vermelho (255); o que separa é quanto verde vem junto (laranja tem 3 a 4 vezes mais). Como a webcam muda o brilho sozinha, o limite não é fixo: fica no meio do caminho entre o centro vermelho e o centro laranja de cada leitura, e só decide quando a peça está claramente de um lado. Repetindo 12 leituras reais, os adesivos corrigidos caíram de 55 para 29.

Para medir: `python tools/estresse_cores.py 300` simula um cubo com vermelho e laranja próximos sob luz quente e exposição variando por face. Antes dessas mudanças: 27% dos cubos lidos 100% certos (72% com correção). Depois: 96% (100% com correção).
