# Sistema Inteligente de Monitoramento Agrícola 🌱🤖

> Um sistema de baixo custo baseado em IoT e Visão Computacional para monitoramento de microclima e detecção automática de pragas (*Trialeurodes vaporariorum*).

![Status do Projeto](https://img.shields.io/badge/Status-Em_Desenvolvimento-yellow)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8s-green)
![ESP32](https://img.shields.io/badge/Hardware-ESP32-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📋 Sobre o Projeto

Este projeto é parte do Trabalho de Conclusão de Curso (TCC) em Engenharia da Computação na **FHO - Fundação Hermínio Ometto**.
O objetivo é desenvolver e validar um protótipo funcional para auxiliar pequenos produtores na tomara de decisão, utilizando conceitos de **Agricultura 4.0**.

O sistema resolve problemas de monitoramento manual (trabalhoso e impreciso) através da automação em três frentes principais: sensoriamento ambiental, visão computacional para contagem de pragas e visualização web centralizada.

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

### 1. Módulo 1: Nó de Sensores IoT (`/Modulo_1`)

Responsável pela coleta de dados vitais do solo e ambiente.
O firmware foi desenvolvido para **ESP32 (NodeMCU)** e comunica-se via protocolo **MQTT**.

* **Hardware Principal:** ESP32.
* **Sensores Integrados:**

  * Umidade do Solo (Capacitivo)
  * BME280
  * BH1750
  * Sensores de pH e EC

<div align="center">
<img src="assets/20251205_100210.jpg" alt="Protótipo do Nó de Sensores" width="600"/>
<p><em>Figura 1: Protótipo funcional do Módulo 1 com ESP32 e sensores.</em></p>
</div>

---

### 2. Módulo 2: Inteligência Artificial (`/Modulo_2`)

Focado na detecção automática da **Mosca-Branca-de-Estufa** (*Trialeurodes vaporariorum*).

* **Modelo:** YOLOv8s
* **Dataset:** IP102 (subconjunto curado)
* **Treinamento:**

  * Resolução 512×512
  * AdamW
  * Mosaic desativado

<div align="center">
<img src="assets/val_batch1_pred.jpg" alt="Detecção de Mosca Branca com YOLOv8" width="800"/>
<p><em>Figura 2: Validação do modelo em diferentes cenários de densidade e iluminação.</em></p>
</div>


## 📊 Resultados do Modelo YOLOv8

| Métrica  | Valor      |
| -------- | ---------- |
| Precisão | **89.61%** |
| Recall   | 67.93%     |
| mAP@0.5  | 75.46%     |
| Latência | 13.98 ms   |

---

### 3. Sistema de Monitoramento (`/Sistema_Monitoramento`)

Dashboard web em tempo real via MQTT.

<div align="center">
<img src="assets/dashboard.png" alt="Dashboard Web" width="800"/>
<p><em>Figura 3: Interface exibindo dados ambientais em tempo real.</em></p>
</div>

---

## 👨‍💻 Autor

**Victor Augusto de Oliveira**
FHO – Engenharia da Computação
📩 [victoroliveira855@alunos.fho.edu.br](mailto:victoroliveira855@alunos.fho.edu.br)