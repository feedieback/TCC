# Sistema Inteligente de Monitoramento Agrícola 🌱🤖

> Um sistema de baixo custo baseado em IoT e Visão Computacional para monitoramento de microclima e detecção automática de pragas (*Trialeurodes vaporariorum*).

![Status do Projeto](https://img.shields.io/badge/Status-Em_Desenvolvimento-yellow)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8s-green)
![ESP32](https://img.shields.io/badge/Hardware-ESP32-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📋 Sobre o Projeto

Este projeto integra o Trabalho de Conclusão de Curso (TCC) em Engenharia da Computação na **FHO - Fundação Hermínio Ometto**. O objetivo central é desenvolver uma solução acessível de **Agricultura 4.0** para auxiliar pequenos produtores na tomada de decisão.

⚠️ **Estado Atual de Desenvolvimento:**
É importante ressaltar que o projeto encontra-se em seus **primeiros estágios de desenvolvimento**. Embora a arquitetura geral esteja definida e os algoritmos de IA validados, a integração de hardware ainda está em fase de prova de conceito. Especificamente, o **Módulo 1** (Nó de Sensores) opera atualmente como um **protótipo de bancada**, focado na validação da leitura dos sensores e da telemetria via MQTT, antes de avançar para a confecção da placa de circuito impresso (PCB) final e encapsulamento para campo.

O sistema visa resolver problemas de monitoramento manual através de três frentes: sensoriamento ambiental, visão computacional para contagem de pragas e visualização web centralizada.

---

## 🏗️ Arquitetura do Sistema

O repositório está organizado conforme os módulos funcionais do projeto:

```text
📁 Raiz
├── 📂 Modulo_1/Monitoramento_Jardim  # Firmware e códigos do Nó de Sensores IoT (ESP32)
├── 📂 Modulo_2/IA                    # Scripts de treinamento e validação da CNN (YOLOv8)
├── 📂 Sistema_Monitoramento          # Plataforma Web (Dashboard/Backend)
└── 📄 Artigo.pdf                     # Documentação acadêmica completa
```

### 1\. Módulo 1: Nó de Sensores IoT (`/Modulo_1`)

Responsável pela coleta de dados vitais do solo e ambiente. Atualmente em versão de **protótipo**, o firmware foi desenvolvido para **ESP32 (NodeMCU)** e comunica-se via protocolo **MQTT**.

  * **Hardware Principal:** ESP32.
  * **Sensores Integrados:**
      * Umidade do Solo (Capacitivo)
      * BME280 (Clima)
      * BH1750 (Luminosidade)
      * Sensores de pH e EC

\<div align="center"\>
\<img src="assets/20251205\_100210.jpg" alt="Protótipo do Nó de Sensores" width="600"/\>
\<p\>\<em\>Figura 1: Protótipo funcional do Módulo 1 em bancada com ESP32 e sensores.\</em\>\</p\>
\</div\>

-----

### 2\. Módulo 2: Inteligência Artificial (`/Modulo_2`)

Focado na detecção automática da **Mosca-Branca-de-Estufa** (*Trialeurodes vaporariorum*).

  * **Modelo:** YOLOv8s
  * **Dataset:** IP102 (subconjunto curado)
  * **Treinamento:**
      * Resolução 512×512
      * AdamW
      * Mosaic desativado

\<div align="center"\>
\<img src="assets/val\_batch1\_pred.jpg" alt="Detecção de Mosca Branca com YOLOv8" width="800"/\>
\<p\>\<em\>Figura 2: Validação do modelo em diferentes cenários de densidade e iluminação.\</em\>\</p\>
\</div\>

## 📊 Resultados do Modelo YOLOv8

| Métrica  | Valor      |
| -------- | ---------- |
| Precisão | **89.61%** |
| Recall   | 67.93%     |
| mAP@0.5  | 75.46%     |
| Latência | 13.98 ms   |

-----

### 3\. Sistema de Monitoramento (`/Sistema_Monitoramento`)

Dashboard web para visualização dos dados em tempo real via MQTT.

\<div align="center"\>
\<img src="assets/dashboard.png" alt="Dashboard Web" width="800"/\>
\<p\>\<em\>Figura 3: Interface exibindo dados ambientais em tempo real.\</em\>\</p\>
\</div\>

-----

## 👨‍💻 Autor

**Victor Augusto de Oliveira**
FHO – Engenharia da Computação
📩 [victoroliveira855@alunos.fho.edu.br](mailto:victoroliveira855@alunos.fho.edu.br)

```