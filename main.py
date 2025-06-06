import cv2
import mediapipe as mp

wCam, hCam = 1200, 480

#Iniciando a captura de vídeo
webcam = cv2.VideoCapture(0)
webcam.set(3, wCam)
webcam.set(4, hCam)

# Iniciando o módulo de desenho do mediapipe
hands = mp.solutions.hands
hand_quantidade = hands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

def classificar_gesto(landmarks):
    # Exemplo: se a ponta do dedo indicador (8) está acima da base (6), dedo está aberto
    dedos_abertos = 0
    # Pontos dos dedos: polegar(4), indicador(8), médio(12), anelar(16), mindinho(20)
    pontas = [4, 8, 12, 16, 20]
    bases = [2, 6, 10, 14, 18]
    for ponta, base in zip(pontas[1:], bases[1:]):  # Ignora o polegar para simplificar
        if landmarks[ponta].y < landmarks[base].y:
            dedos_abertos += 1
    if dedos_abertos == 0:
        return "Punho Fechado"
    elif dedos_abertos == 4:
        return "Mao Aberta" 
    else:
        return "Outro Gesto" 


# Para armazenar a posição anterior das duas mãos
posicoes_x_anteriores = [None, None]
limiar = 0.06  # Ajuste para sensibilidade
contadores_aceno = [0, 0]  # Contador para cada mão
frames_para_manter = 30    # Quantos frames manter o texto após detectar aceno

while True:
    sucesso, img = webcam.read()
    if not sucesso:
        print("Webcam não encontrada ou ocupada! Feche outros programas que usam a webcam e tente novamente.")
        cv2.waitKey(3000)  # Espera 3 segundos para o usuário ver a mensagem
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = hand_quantidade.process(imgRGB)

    pedido_ajuda = False

    # Inicializa textos para cada lado
    texto_direita = ""
    texto_direita_mov = ""
    texto_esquerda = ""
    texto_esquerda_mov = ""

    if resultados.multi_hand_landmarks:
        movimentos = []
        for idx, mao in enumerate(resultados.multi_hand_landmarks[:2]):
            mpDraw.draw_landmarks(img, mao, hands.HAND_CONNECTIONS)
            gesto = classificar_gesto(mao.landmark)

            # Centro da mão
            x_centro = sum([lm.x for lm in mao.landmark]) / len(mao.landmark)

            movimento = ""
            if posicoes_x_anteriores[idx] is not None:
                if abs(x_centro - posicoes_x_anteriores[idx]) > limiar:
                    contadores_aceno[idx] = frames_para_manter  # Reinicia o contador
            posicoes_x_anteriores[idx] = x_centro

            # Diminui o contador a cada frame
            if contadores_aceno[idx] > 0:
                movimento = "Acenando"
                contadores_aceno[idx] -= 1
            else:
                movimento = ""

            movimentos.append(movimento)

            # Identifica se é mão direita ou esquerda
            if resultados.multi_handedness:
                label = resultados.multi_handedness[idx].classification[0].label
                if label == "Right":
                    texto_direita = f"Mao Esquerda: {gesto}"
                    texto_direita_mov = movimento
                else:
                    texto_esquerda = f"Mao Direira: {gesto}"
                    texto_esquerda_mov = movimento

        # Se duas mãos e ambas acenando, mostra pedido de ajuda
        if len(movimentos) == 2 and movimentos[0] == "Acenando" and movimentos[1] == "Acenando":
            pedido_ajuda = True

    # Exibe os textos nos lados opostos da tela
    cv2.putText(img, texto_direita, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, texto_direita_mov, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (139, 0, 0), 2)
    cv2.putText(img, texto_esquerda, (wCam - 550, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, texto_esquerda_mov, (wCam - 550, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (139, 0, 0), 2)

    # Se pedido de ajuda, exibe mensagem centralizada
    if pedido_ajuda:
        texto = "PEDIDO DE AJUDA!"
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        escala = 2
        espessura = 5
        (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte, escala, espessura)
        x = (wCam - largura_texto) // 2
        y = hCam - 40 
        cv2.putText(img, texto, (x, y), fonte, escala, (0,0,255), espessura)

    cv2.imshow("Maos", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()