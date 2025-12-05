# Sistema Inteligente de Monitoramento Agrícola 🌱🤖

> Um sistema de baixo custo baseado em IoT e Visão Computacional para monitoramento de microclima e detecção automática de pragas (*Trialeurodes vaporariorum*).

![Status do Projeto](https://img.shields.io/badge/Status-Em_Desenvolvimento-yellow)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8s-green)
![ESP32](https://img.shields.io/badge/Hardware-ESP32-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📋 Sobre o Projeto

Este projeto é parte do Trabalho de Conclusão de Curso (TCC) em Engenharia da Computação na **FHO - Fundação Hermínio Ometto**. [cite_start]O objetivo é desenvolver e validar um protótipo funcional para auxiliar pequenos produtores na tomara de decisão, utilizando conceitos de **Agricultura 4.0**[cite: 400, 402, 430].

[cite_start]O sistema resolve problemas de monitoramento manual (trabalhoso e impreciso) através da automação em três frentes principais: sensoriamento ambiental, visão computacional para contagem de pragas e visualização web centralizada[cite: 409, 411].

---

## 🏗️ Arquitetura do Sistema

O repositório está organizado conforme os módulos funcionais do projeto:

```text
📁 Raiz
├── 📂 Modulo_1/Monitoramento_Jardim  # Firmware e códigos do Nó de Sensores IoT (ESP32)
├── 📂 Modulo_2/IA                    # Scripts de treinamento e validação da CNN (YOLOv8)
├── 📂 Sistema_Monitoramento          # Plataforma Web (Dashboard/Backend)
└── 📄 Artigo.pdf                     # Documentação acadêmica completa
````

### 1\. Módulo 1: Nó de Sensores IoT (`/Modulo_1`)

Responsável pela coleta de dados vitais do solo e ambiente. [cite_start]O firmware foi desenvolvido para **ESP32 (NodeMCU)** e comunica-se via protocolo **MQTT**[cite: 468, 469].

  * **Hardware Principal:** ESP32.
  * **Sensores Integrados:**
      * Umidade do Solo (Capacitivo).
      * BME280 (Temperatura, Umidade e Pressão).
      * BH1750 (Luminosidade).
      * [cite_start]Sensores de pH e Condutividade Elétrica (EC)[cite: 468].

\<div align="center"\>
\<img src="20251205\_090132.jpg" alt="Protótipo do Nó de Sensores" width="600"/\>
\<p\>\<em\>Figura 1: Protótipo funcional do Módulo 1 com ESP32 e sensores.\</em\>\</p\>
\</div\>

### 2\. Módulo 2: Inteligência Artificial (`/Modulo_2`)

Focado na detecção automática da **Mosca-Branca-de-Estufa** (*Trialeurodes vaporariorum*) em armadilhas adesivas amarelas. [cite_start]Utiliza a arquitetura **YOLOv8s** otimizada para objetos pequenos[cite: 7, 40].

  * **Modelo:** YOLOv8s (Small).
  * **Dataset:** Subconjunto curado do IP102.
  * **Estratégia de Treino:**
      * Resolução: 512x512 pixels.
      * Otimizador: AdamW.
      * [cite_start]*Data Augmentation*: Mosaic desativado para preservar detalhes de pequenos objetos[cite: 9, 66, 70].
  * **Resultados Obtidos:**
      * [cite_start]**Precisão:** 89.61% (Minimização de falsos positivos)[cite: 81].
      * [cite_start]**mAP@0.5:** 75.46%[cite: 81].
      * [cite_start]**Velocidade:** \~71 FPS (13.98 ms) em GPU[cite: 83].

\<div align="center"\>
\<img src="val\_batch1\_pred.jpg" alt="Detecção de Mosca Branca com YOLOv8" width="800"/\>
\<p\>\<em\>Figura 2: Validação do modelo em diferentes cenários de densidade e iluminação.\</em\>\</p\>
\</div\>

### 3\. Sistema de Monitoramento (`/Sistema_Monitoramento`)

Dashboard web para visualização em tempo real dos dados coletados pelos módulos.

  * **Protocolo:** MQTT.
  * [cite_start]**Funcionalidades:** Gráficos históricos, alertas e visualização de feeds (Temperatura, Umidade, Contagem de Pragas)[cite: 475, 489].

\<div align="center"\>
\<img src="Captura de tela 2025-12-05 085925.png" alt="Dashboard Web" width="800"/\>
\<p\>\<em\>Figura 3: Interface do usuário exibindo dados ambientais em tempo real.\</em\>\</p\>
\</div\>

-----

## 🚀 Como Executar

### Pré-requisitos

  * Python 3.8+
  * Bibliotecas: `ultralytics`, `opencv-python`, `pandas`, `torch` (ver `requirements.txt` em cada módulo).
  * Hardware: ESP32 e Raspberry Pi (para deploy em borda).

### Instalação e Uso (IA)

1.  Navegue até a pasta da IA:
    ```bash
    cd Modulo_2/IA
    ```
2.  Instale as dependências:
    ```bash
    pip install ultralytics
    ```
3.  Para rodar a inferência em uma imagem de teste:
    ```python
    from ultralytics import YOLO
    model = YOLO('best.pt') # Utilize os pesos treinados
    results = model('caminho/para/imagem.jpg')
    ```

-----

## 📊 Resultados e Performance

O modelo de IA demonstrou robustez em cenários de alta densidade e oclusão, superando limitações da contagem manual.

| Métrica | Valor | Descrição |
| :--- | :--- | :--- |
| **Precisão** | **89.61%** | [cite_start]Alta confiabilidade para tomada de decisão no MIP[cite: 124]. |
| **Recall** | 67.93% | [cite_start]Impactado por oclusões severas em folhas densas[cite: 125]. |
| **mAP@0.5** | 75.46% | [cite_start]Equilíbrio geral de detecção[cite: 81]. |
| **Latência** | 13.98 ms | [cite_start]Viável para processamento em tempo real (Edge Computing)[cite: 83]. |

-----

## 👨‍💻 Autor

**Victor Augusto de Oliveira**

  * **Instituição:** FHO - Fundação Hermínio Ometto
  * **Curso:** Engenharia da Computação
  * **Contato:** [victoroliveira855@alunos.fho.edu.br](mailto:victoroliveira855@alunos.fho.edu.br)

-----

## 📄 Referências

O embasamento teórico e os resultados detalhados podem ser encontrados nos documentos anexados ao repositório ou nas referências abaixo:

  * OLIVEIRA, V. A. *Sistema Inteligente de Monitoramento Agrícola*. [cite_start]TCC, FHO, 2025[cite: 399].
  * OLIVEIRA, V. A. *Detecção Automática de Trialeurodes vaporariorum em Cultivos de Tomate Utilizando Visão Computacional*. [cite_start]Artigo Científico, 2025[cite: 1, 3].
  * KHAN, A. et al. *AI-Enabled Crop Management Framework...*. [cite_start]Plants, 2024[cite: 24].

<!-- end list -->

```
```