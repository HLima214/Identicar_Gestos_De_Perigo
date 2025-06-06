# 🌍 **ONTRAK** - Pulseira Inteligente para Situações de Emergência

## 🧭 Visão Geral

Este projeto propõe uma **pulseira inteligente com função de vibração**, voltada para **comunicação emergencial em situações de desastre**, como apagões, enchentes, deslizamentos e terremotos. A proposta atende ao tema da *Global Solution 2025* do primeiro semestre, com foco em acessibilidade, resiliência e sustentabilidade.

## 🚨 Problema

Eventos extremos frequentemente afetam milhões de pessoas, deixando comunidades inteiras isoladas, sem energia elétrica, sinal de celular ou acesso a serviços essenciais. Isso é especialmente grave para grupos vulneráveis, como:

- 👵 **Idosos**, com dificuldade de locomoção ou uso de tecnologia;
- ♿ **Pessoas com deficiência**, que necessitam de assistência especializada;
- 🏚️ **Pessoas em situação de pobreza**, que vivem em áreas de risco e têm menos recursos.

Essas pessoas, em muitos casos, **não conseguem pedir ajuda de forma eficaz**, o que agrava ainda mais o impacto dessas ocorrências.

## 💡 Solução Proposta

A solução consiste em uma **pulseira inteligente** que:

- **Recebe alertas** por meio de **vibração**;
- Exibe **ícones e cores** que indicam o tipo e a gravidade do desastre;
- Permite o **envio de pedidos de socorro** com um **gesto simples** (ex: dois toques rápidos);
- **Transmite localização** do usuário para uma **Central de Monitoramento** ou **parente monitorando remotamente através do aplicativo**.

Essa abordagem visa **garantir comunicação mesmo em contextos de falha elétrica ou tecnológica**, como em **elevadores sem energia** ou áreas com **infraestrutura comprometida**.

## 📡 Comunicação e Transmissão de Dados

Para manter os custos acessíveis e evitar tecnologias de alto custo (como satélites ou chips celulares), foi adotada a seguinte arquitetura de comunicação:

###  Torres de Comunicação Inteligentes

Infraestrutura equipada com:

- **Receptores de sinal da pulseira**;
- **Câmeras de alta precisão** para mapear áreas e identificar gestos de socorro.

### **Para vídeo de demonstração da câmera [clique aqui](https://youtu.be/BPJhwVmu840)**

### **Para acessar o código fonte [clique aqui](https://github.com/HLima214/Identicar_Gestos_De_Perigo/blob/main/main.py)**

Essas torres funcionarão em **parceria com a Defesa Civil** e empresas de telecomunicação, que cederão parte de sua rede para a operação do sistema.

### 🔄 Fluxo de Comunicação

```text
Pulseira → Torres com câmeras → Central de Monitoramento / Aplicativo
```

## 📱 Aplicativo Integrado

Um aplicativo complementar **replica** as **funções da pulseira** e adiciona um painel de **monitoramento** em **tempo real**, permitindo:

- Visualização da **localização do usuário**;

- Histórico de alertas;

- **Acompanhamento** por familiares, cuidadores ou autoridades.

Essa interface garante uma experiência acessível e inclusiva, especialmente para pessoas que possuem parentes que se encaixam no grupo de pessoas vulneráveis e querem monitorar para garantir sua segurança

## 🧭 Central de Resgate e Monitoramento
A Central de Monitoramento será integrada aos serviços de emergência públicos (como Defesa Civil, SAMU e Bombeiros) e oferecerá:

- Análise em tempo real dos alertas;

- Coordenação das equipes de resgate;

- Painéis interativos e geolocalização de vítimas;

- Equipes treinadas para resposta rápida.

## 🔋 Energia e Sustentabilidade

### ⚡ Fonte de energia
Para garantir funcionamento contínuo mesmo sem energia elétrica, foram avaliadas diferentes soluções:

🔌 USB ou troca de baterias (viáveis, porém limitadas);

☀️ Carregamento solar (escolhida):

- Placas solares de 3W a 6W com saída USB, custando entre R$ 36,00 a R$ 110,00;

- Sustentável, econômica e confiável.

❌ Energia cinética foi descartada por alto custo e complexidade técnica.

## 🌐 Impacto Social
Essa solução busca reduzir a mortalidade e aumentar a resposta emergencial em regiões críticas, com foco em:

- Inclusão social;

- Comunicação acessível e simplificada;

- Uso racional de recursos públicos e privados.

## 🤝 Parcerias Estratégicas
- Defesa Civil

- Corpo de Bombeiros

- Empresas de Telecomunicação

- ONGs de assistência social

- Fabricantes de dispositivos eletrônicos sustentáveis

## 📌 Conclusão
A pulseira inteligente representa uma inovação acessível, escalável e socialmente transformadora. Combinando tecnologia de baixo custo, integração com serviços públicos e design inclusivo, ela oferece uma nova forma de salvar vidas em desastres, promovendo resiliência comunitária e justiça social.

## 👥 Integrantes do Grupo
Henrique Lima | RM551528

Anny Carolina Andrade Dias | RM98295

Sofia Amorim Coutinho | RM552534

---

_Desenvolvido para a Global Solution 2025 — 1º semestre._
