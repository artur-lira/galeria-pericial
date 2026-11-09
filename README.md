# Galeria Pericial

Programa para Windows que abre uma pasta de imagens, com subpastas, e permite **visualizar em galeria, buscar e encontrar imagens semelhantes**: cópias, miniaturas e caches, versões redimensionadas, rotacionadas ou espelhadas, **recortes** e versões **editadas**. Foi pensado para o trabalho pericial: os originais são apenas lidos, todos os arquivos têm hash e as exportações são conferidas.

Atende coleções de **centenas de milhares de imagens** e usa uma ou **várias GPUs NVIDIA (CUDA)** quando houver.

> **Aviso:** os resultados são **indicativos** (triagem). Eles não substituem o exame pericial nem a fixação dos achados em laudo.

---

## Funcionalidades

| Área | O que faz |
|---|---|
| **Galeria** | Miniaturas com tamanho ajustável, prévia, visualizador com zoom e selos (≈ semelhantes, **!** miniatura EXIF divergente, **GPS**, **EXT** extensão divergente) |
| **Formatos** | JPG, PNG, GIF, BMP, TIFF, WEBP, HEIC/HEIF/AVIF, RAW de câmeras (CR2, CR3, NEF, ARW, DNG…), PSD, ICO, JPEG 2000, EXR/HDR e outros |
| **Formato real** | Reconhece o formato pela assinatura do arquivo (*magic bytes*). Encontra imagens **sem extensão** ou **renomeadas** (`.dat`, `.tmp`…) e aponta extensão divergente |
| **Busca** | Por nome, caminho e **todos os metadados** (EXIF, GPS, XMP). Vários termos combinados com E, e `-termo` para excluir |
| **Filtros** | Período (data de captura ou de modificação), formato, com GPS, sem EXIF, duplicatas SHA-256, miniatura EXIF divergente, imagens uniformes, erros de leitura |
| **Semelhantes** | Cada imagem é comparada com **todos** os arquivos da pasta e subpastas. A lista vem em ordem de data de modificação e mostra confiança, tamanho e o tipo de relação |
| **Comparação** | Lado a lado com a **região do recorte destacada**, **mapa de diferenças** após alinhamento, sobreposição com transparência e metadados confrontados |
| **Análise global** | Compara todas as imagens com todas (multi-GPU) e marca quais têm semelhantes |
| **Relatório** | DOCX (editável) e PDF das imagens selecionadas, com ficha, hashes, tabela de semelhantes e figuras |
| **Exportação pericial** | Copia a imagem e as semelhantes **sem renomear nada**, preservando estrutura de pastas e datas, com SHA-256/MD5 conferidos e manifesto |
| **Diagnóstico** | Uso de cada GPU, teste de carga e perfil da indexação com o gargalo provável (CPU, disco, gravação ou GPU) |

---

## Requisitos

| Item | Mínimo / recomendado |
|---|---|
| Sistema | Windows 10 ou 11, 64 bits |
| Para gerar o executável | Python 3.11 a 3.13 (64 bits) e internet, só durante o build |
| GPU (opcional) | NVIDIA com driver atualizado. O CUDA já vem com o programa. Testado para 1 a 3 RTX A4000 (16 GB) |
| Disco | ~5 GB para o programa (versão CUDA). O índice ocupa ~15 KB por imagem (≈ 4–5 GB para 300 mil) |
| RAM | 16 GB ou mais para coleções grandes em modo CPU |

---

## Instalação (gerar o `.exe`)

1. Instale o **Python 3.11–3.13 (64 bits)** pelo [python.org](https://www.python.org/downloads/windows/), marcando **"Add python.exe to PATH"**.
2. Baixe o código, com **Code → Download ZIP** nesta página ou com:
   ```bash
   git clone https://github.com/artur-lira/galeria-pericial
   ```
3. Dê dois cliques em **`build.bat`**, que gera a versão com GPU NVIDIA. Para a versão só com CPU, no Prompt de Comando:
   ```bash
   build.bat cpu
   ```
4. O executável fica em **`dist\GaleriaPericial\GaleriaPericial.exe`**. Copie a pasta `dist\GaleriaPericial` inteira para as estações; depois de gerada, ela funciona **sem internet** e sem Python.

O `build.bat`:
- cria um ambiente isolado (`.venv`);
- instala as dependências e o PyTorch (CUDA ou CPU);
- baixa os pesos públicos da ResNet-50/ImageNet (~100 MB, com verificação de hash);
- roda um autoteste do motor;
- empacota o programa com o PyInstaller.

Para rodar direto do código, sem gerar o `.exe`, use `executar.bat` (depois de rodar o `build.bat` uma vez).

---

## Uso rápido

1. **Abrir pasta** (Ctrl+O). O programa indexa todos os arquivos, inclusive os das subpastas.
2. **Análise global** (opcional): marca na galeria as imagens que têm semelhantes (selo ≈N).
3. **Clique em uma imagem**: a aba *Semelhantes* lista as relações encontradas em toda a pasta. Um duplo clique abre a comparação.
4. **Ctrl+R** gera o relatório das imagens selecionadas; **Ctrl+E** exporta as imagens selecionadas junto com as semelhantes.

| Etapa | O que faz | Quando |
|---|---|---|
| **Indexação** | Lê cada arquivo uma vez: hashes, metadados, formato real, miniatura, pHash, regiões uniformes e descritores da CNN. É **incremental** (só reprocessa o que mudou) e **retomável** | Ao abrir a pasta ou com F5 |
| **Análise global** | Todas × todas, com conferência; alimenta os selos ≈ e o filtro "Com semelhantes" | Botão *Análise global* |
| **Busca individual** | Uma × todas, com classificação final, localização do recorte e mapa de diferenças | Ao clicar na imagem |

---

## Como a semelhança é detectada

| Camada | Técnica | Detecta |
|---|---|---|
| 1 | **SHA-256** | Cópias idênticas, bit a bit |
| 2 | **pHash** (DCT) nas 8 orientações + conferência de conteúdo | Versões redimensionadas, recomprimidas, miniaturas e caches, rotações e espelhamentos |
| 3 | **CNN regional** (ResNet-50, 15 regiões e escalas por imagem) + busca vetorial em GPU | Triagem de recortes e edições entre centenas de milhares de imagens |
| 4 | **ORB + RANSAC** (homografia) | Confirmação geométrica: recortes de qualquer proporção (inclusive zoom central), ampliados ou girados, e localização do recorte |
| 5 | **Diferença após alinhamento** | Alterações locais: inserções, remoções, tarjas |

Proteções contra falsos positivos:
- **Fundos transparentes, margens e "moldes" iguais:** o resultado do pHash é conferido pelo conteúdo, sem contar o fundo.
- **Imagens de cor sólida ou quase vazias:** só são comparadas por SHA-256.
- **Regiões uniformes:** fundos e céu liso não entram na comparação da CNN.
- **Cores:** pHash próximo só vale como "mesma imagem" se as cores médias forem compatíveis.

---

## Integridade pericial

- **Somente leitura:** nenhum arquivo da pasta analisada é alterado, e nada é criado dentro dela. O índice fica em `%LOCALAPPDATA%\GaleriaPericial\indices\`.
- **SHA-256 e MD5** de todos os arquivos ficam no índice, nos CSVs, no relatório e na exportação.
- **Exportação:**
  - nomes originais e estrutura de subpastas preservados;
  - datas de criação e modificação preservadas;
  - nada é sobrescrito;
  - o hash do original é recalculado no momento da exportação e comparado com o índice e com a cópia;
  - são gerados `manifesto.csv`, `hashes_sha256.txt` e `hashes_md5.txt`, conferíveis com `certutil`, `Get-FileHash` ou `sha256sum -c`.
- **Relatório:** o SHA-256 dos arquivos DOCX e PDF gerados é exibido ao final.

---

## Limitações conhecidas

- Recortes muito pequenos (menos de ~5% da área) ou de regiões sem textura podem não ser detectados.
- Os escores de confiança são **heurísticos**, não probabilidades. O limiar dos selos ≈ (padrão 0,70) deve ser calibrado com casos conhecidos da coleção.
- Sem a CNN (PyTorch ausente ou falhando), recortes só são procurados por força bruta em coleções de até 5.000 imagens.
- Com 2 ou 3 GPUs, a indexação costuma ser limitada pela CPU e pelo disco (leitura, hash e decodificação), e não pelas placas.
- Os caches de miniaturas do Windows (`Thumbs.db`, `thumbcache_*.db`) não são lidos diretamente.

---

## Estrutura do código

```
main.py                 ponto de entrada
galeria/
  ui.py                 interface (PySide6): galeria, filtros, abas, comparação, diagnóstico de GPUs
  store.py              índice SQLite/FTS5 por pasta, indexação paralela e incremental
  loader.py             leitura dos formatos, metadados (EXIF/XMP/GPS), miniatura EXIF embutida
  formats.py            identificação do formato pela assinatura (magic bytes)
  similarity.py         pHash, ORB/RANSAC, classificação, conferência de conteúdo, mapa de diferenças
  engine.py             CNN regional e busca vetorial em CPU, GPU ou multi-GPU
  search.py             busca de semelhantes (uma × todas) e análise global (todas × todas)
  report.py             relatório DOCX e PDF
  export.py             exportação pericial com manifesto e conferência de hashes
build.bat               gera o executável (CUDA ou CPU)
executar.bat            executa a partir do código-fonte
```

O manual completo, com todas as opções, os ajustes de GPU e os detalhes do método, está em **[LEIAME.md](LEIAME.md)**.

---

## Licença

[MIT](LICENSE). Copyright (c) 2026 Artur Lira dos Santos.
