# Universidade Federal de São Paulo (UNIFESP)
## Disciplina: Inteligência Artificial
**Atividade:** 16  
**Aluno:** João Vitor Mâncio Chaves  
**RA:** 176.534  

---

# Classificação de Flores com Redes Neurais Convolucionais (CNN) e Grad-CAM

## Resumo
Este projeto tem como objetivo a construção, treinamento e validação de um modelo de base convolucional (CNN) utilizando Transfer Learning com a arquitetura **MobileNetV2** para a classificação de 5 espécies de flores (Daisy, Dandelion, Roses, Sunflowers, Tulips). Além da predição, o projeto implementa o **Grad-CAM** (Gradient-weighted Class Activation Mapping), um mecanismo de atenção visual que permite analisar as regiões de interesse (ROIs) que o modelo utiliza para as suas tomadas de decisão.

## Funcionalidades e Escopo
- **Data Pipeline:** Carregamento automatizado de dados do Kaggle (`rahmasleam/flowers-dataset`) utilizando a biblioteca `kagglehub`. Efetua Data Augmentation (Flip, Rotation, Zoom) visando maior robustez no treinamento.
- **Modelagem:** Aplicação de Transfer Learning sobre o base model de pesos do _ImageNet_.
- **Avaliação e Métricas:** Extração multivariada das validações de Loss, Accuracy, Matriz de Confusão e Curvas ROC/AUC multiclasse.
- **Explicabilidade:** Mapas de calor gerados com Grad-CAM para interpretabilidade semântica botânica.

---

## 🚀 Como executar o projeto

### 1. Clonar o repositório
```bash
git clone https://github.com/JoaoVitorChaves-05/CNN-Flowers.git
cd CNN-Flowers
```

### 2. Criar e ativar o Ambiente Virtual (venv)

**No Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Nota: Se houver erro de permissão no Windows, execute `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` antes de ativar).*

**No Linux ou macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as Dependências
Com o ambiente ativado, instale as bibliotecas necessárias contidas no `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Executar o Projeto
O ponto de entrada principal que compila os conjuntos de dados, treina a CNN (ou carrega pesos salvos), dispara os testes estáticos e salva os gráficos na pasta de output é:
```bash
python index.py
```

Os resultados gerados estarão localizados nas pastas internas (ex: relatórios estatísticos e imagens Grad-CAM em `output/gradcam`).