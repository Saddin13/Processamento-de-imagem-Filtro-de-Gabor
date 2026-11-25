# Filtro de Gabor — Vitor Saddi · PDI

Página local (Streamlit) que mostra, passo a passo, a aplicação do filtro de Gabor:

1. imagem original
2. tons de cinza (`Y = 0,299 R + 0,587 G + 0,114 B`)
3. kernel de Gabor, calculado pela fórmula matemática (em NumPy)
4. imagem filtrada (parte real ou magnitude)
5. (opcional) conferência do kernel com `cv2.getGaborKernel`

Os parâmetros (λ, θ, ψ, σ, γ e tamanho do kernel) ficam na barra lateral e tudo é recalculado ao mexer.

## Instalar (uma vez, com internet — faça em casa)

```bash
pip install -r requirements.txt
```

## Rodar

```bash
streamlit run app.py
```

Abre em `http://localhost:8501`. Roda inteiro no seu computador, sem internet.
Se não abrir sozinho, copie esse endereço no navegador.

## Dicas para o projetor

- `Ctrl` + `+` / `Ctrl` + `-` ajusta o zoom da página; `F11` coloca em tela cheia.
- A barra lateral pode ser recolhida pela setinha no canto para dar mais espaço às imagens.
- Sem imagem enviada, a página abre com uma imagem de teste (listras em 4 orientações).

## Testes

```bash
python -m pytest
```

## Observação sobre o OpenCV

O OpenCV é **opcional** (só serve para a conferência do passo 5). Em computadores com política
de segurança que bloqueia o `opencv-python`, o app funciona normalmente e apenas avisa que a
conferência está indisponível. Para ativá-lo: `pip install opencv-python`.
