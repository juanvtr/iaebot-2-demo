# Roteiro da demo — IAEbot 2.0

## 1. Contexto
Explique que o IAEbot 1.0 validou o RAG textual, mas não interpretava adequadamente conhecimento presente em gráficos, tabelas e imagens.

## 2. Pergunta principal
**Qual foi a deformação máxima registrada no ensaio de tração, conforme o gráfico da página 12?**

Resposta esperada: **14,2%**.

Na tela, destaque:
- a resposta;
- a fonte `Relatorio_Tecnico_Demo_IAEbot_2_0.pdf`;
- a página 12;
- o tipo de evidência `image`;
- o score de similaridade.

## 3. Comparação proposital
A tabela da página 10 contém valores até **13,3%**, mas a pergunta pede especificamente o valor **conforme o gráfico da página 12**. O sistema deve priorizar a evidência visual da Figura 4 e responder **14,2%**.

## 4. Fechamento
Explique que esta é uma prova de conceito reprodutível: a evidência visual foi pré-processada e armazenada como descrição estruturada. A evolução completa do IAEbot 2.0 prevê OCR, extração automática de tabelas e modelos multimodais locais, como LLaVA, no pipeline de ingestão.
