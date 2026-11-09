# Galeria Pericial: manual completo

Visão geral, instalação e uso rápido estão no [README](README.md). Este manual detalha todas as opções.

Programa para Windows que abre uma pasta (com subpastas) contendo até centenas de milhares de imagens.
Ele permite:
- ver tudo em galeria;
- buscar por nome, metadados e data;
- para cada imagem, encontrar entre **todos os arquivos da pasta e das subpastas** as cópias, as miniaturas e caches, as versões redimensionadas, rotacionadas ou espelhadas, os **recortes** e as versões **editadas**.

Também identifica imagens **sem extensão ou com extensão trocada** pela assinatura do arquivo (magic bytes) e gera **relatório DOCX/PDF** das imagens selecionadas.

## Como gerar o .exe

1. Instale o Python 3.11, 3.12 ou 3.13, 64 bits, pelo site python.org, marcando "Add python.exe to PATH".
2. Execute um destes comandos:
   - `build.bat` gera a versão **GPU NVIDIA (CUDA)**. Ela usa uma GPU ou **várias**, e só exige o driver NVIDIA atualizado.
   - `build.bat cpu` gera a versão somente CPU, que é menor.
3. O resultado fica em `dist\GaleriaPericial\`. Copie a pasta inteira, que já inclui a subpasta `modelos` com os pesos da CNN.
   - Na versão CUDA a pasta ocupa cerca de 4 a 5 GB.
   - Depois de gerada, ela funciona **sem internet**.

O build baixa, uma única vez, os pesos públicos da ResNet-50/ImageNet (`resnet50-0676ba61.pth`, ~100 MB, com verificação de hash). Em seguida, roda um autoteste do motor.

## Processamento e GPU (ajuste automático)

O programa detecta na inicialização quantas GPUs NVIDIA existem e quanta memória cada uma tem. Ele se ajusta a isso sozinho, sem nenhuma configuração.

| Situação | Comportamento |
|---|---|
| 1, 2 ou 3 RTX A4000 (16 GB) | Usa todas. Cada lote da CNN vai para a primeira placa livre, e o índice vetorial é dividido igualmente entre elas (cerca de 100 mil imagens por placa com 3 GPUs) |
| GPUs de tamanhos diferentes | O índice é dividido na proporção da memória de cada placa |
| Uma GPU falha no autoteste | Só ela é descartada, as demais continuam, e a barra de status informa qual saiu |
| Nenhuma GPU funciona | O programa passa para a CPU automaticamente |

Tamanhos de lote para placas de 16 GB como a A4000:
- 96 imagens por lote na CNN;
- na análise global, 128 consultas × 60 mil imagens por bloco, cerca de 1 GB temporário por placa;
- o índice de 300 mil imagens ocupa 2,3 GB de VRAM no total.

### Conferir se todas as GPUs estão trabalhando

- O botão **GPUs** na barra de ferramentas abre o diagnóstico. Ele mostra, para cada placa, a VRAM usada pelo programa, a VRAM total ocupada, o número de imagens processadas pela CNN, os lotes, o tempo ativo e as consultas da análise global. Os números são atualizados a cada segundo.
- O quadro **Perfil da indexação**, na mesma tela, mostra a vazão real (img/s), a ocupação média das GPUs, o tempo gasto em gravação no índice e em espera por GPU, e o tempo médio de cada etapa por imagem (leitura do disco, hash, metadados, decodificação, pHash/miniatura, preparação para a CNN). Com base nisso, indica o **gargalo provável**: CPU, disco, gravação ou GPU.
- O botão **Teste de carga** mede imagens/s de cada GPU sozinha e de todas juntas, e informa o ganho obtido (o ideal com 3 placas é 3×).
- Durante a indexação e a análise global, a barra de status mostra quantas imagens cada GPU já processou.
- O **Gerenciador de Tarefas do Windows não mostra o uso CUDA** no gráfico padrão "3D". Por isso a placa ligada ao monitor parece ser a única em uso. Para ver o uso real, troque um dos gráficos da GPU para "Cuda" ou "Compute_0", ou rode `nvidia-smi -l 1` no Prompt de Comando.

O seletor "Processamento" na barra de ferramentas lista Automático (todas), CPU e todas as combinações de placas (GPU 0 + GPU 1, GPU 2 sozinha etc.). Isso permite deixar uma placa livre para outro trabalho.

Com 2 ou 3 GPUs, quem limita a indexação passa a ser a CPU (leitura do arquivo, SHA-256/MD5, decodificação) e o disco. Por isso, prefira manter o material em SSD/NVMe local durante a indexação.

### Ordem de grandeza

| Etapa | CPU (8-16 núcleos) | GPU |
|---|---|---|
| Indexação sem CNN (hash, metadados, miniatura, pHash) | ~150-300 imagens/s; costuma ser limitada pelo disco | igual |
| Descritores CNN | ~10-20 imagens/s | ~500-1000 imagens/s por A4000 (na prática, limitado pela decodificação na CPU) |
| Busca de semelhantes de 1 imagem x 300 mil | 1-3 s | < 1 s |
| Análise global (todas x todas) de 300 mil | horas | minutos |

A indexação é **incremental e retomável**: Cancelar ou fechar o programa preserva o que já foi feito, e o F5 continua de onde parou. Só os arquivos novos ou alterados são reprocessados.

## Uso

| Ação | Como |
|---|---|
| Abrir pasta | Ctrl+O. Com "Subpastas", a varredura é recursiva |
| Detectar imagens sem extensão | Marcado por padrão: lê a assinatura de **todos** os arquivos, inclusive os sem extensão e os renomeados (`.dat`, `.tmp`...) |
| Busca textual | Procura em nome, caminho e **todos** os metadados (EXIF, GPS, XMP). Vários termos são combinados com E, e `-termo` exclui |
| Filtros | Formato real, com GPS, sem EXIF, miniatura EXIF divergente, duplicatas SHA-256, **extensão divergente ou ausente**, erros |
| Filtro por período | Basta alterar as datas "de/até": o filtro é ativado sozinho. A data padrão é a de captura (EXIF; na falta, XMP/PNG) e, se a imagem não tiver nenhuma, a de modificação do arquivo. Também é possível usar "somente captura" ou "somente modificação". As datas iniciais cobrem toda a coleção, e a barra de status informa quantas imagens ficaram de fora por não ter a data escolhida |
| Limpar filtros | Volta a exibir todas as imagens. Esse é o estado padrão ao iniciar o programa, que reabre automaticamente a última pasta, e ao abrir uma pasta |
| Semelhantes | Ao clicar em uma imagem, ela é comparada com **todos** os arquivos indexados da pasta e das subpastas. As colunas são: imagem, confiança, data de modificação, tamanho e detalhe. **Por padrão, a lista vem em ordem crescente de data de modificação**, com os itens sem data no final. Clicar no título de uma coluna reordena a lista. Por padrão aparece uma lista única em ordem cronológica, com o tipo de relação na coluna Detalhe. Marcando "Agrupar por tipo de relação", os itens são separados por tipo, mantendo a ordem por data dentro de cada grupo |
| Comparar | Duplo clique em uma semelhante abre o lado a lado (com o recorte destacado), o mapa de diferenças, a sobreposição e os metadados confrontados |
| Análise global | Compara todas com todas e alimenta os selos ≈N e o filtro "Com semelhantes". O "Limiar ≈" ajusta a sensibilidade dos selos |
| **Exportar com semelhantes** | Selecione uma ou mais imagens e use **Ctrl+E** (ou o botão na aba Semelhantes). Cada exportação cria uma pasta nova, `exportacao_AAAAMMDD_HHMMSS`, com uma subpasta por imagem de referência. Ver "Exportação pericial" abaixo |
| **Relatório** | Selecione as imagens (Ctrl/Shift+clique) e use **Relatório das selecionadas** (Ctrl+R). São gerados **DOCX (editável) e PDF**, e o SHA-256 de cada arquivo gerado é exibido ao final |

Selos na galeria: **≈N** = semelhantes; **!** = miniatura EXIF divergente; **GPS** = imagem com coordenadas; **EXT** = extensão ausente ou que não corresponde ao conteúdo.

## Exportação pericial

- **Nomes originais preservados:** nenhum arquivo é renomeado. A ordem e a relação com a referência ficam **somente no `manifesto.csv`**.
- **Estrutura preservada:** o caminho relativo à pasta analisada é reproduzido no destino. Por exemplo, `Pasta B\sub\foto.jpg` vira `exportacao_…\001_referencia_foto_jpg\Pasta B\sub\foto.jpg`. Isso mantém a proveniência e evita conflito entre arquivos homônimos de pastas diferentes.
- **Datas preservadas:** modificação e acesso são mantidas, e no Windows também a **data de criação**, que a cópia comum do sistema não preserva. Atributos como somente leitura também são mantidos.
- **Nada é sobrescrito:** cada exportação usa uma pasta nova. Se mesmo assim um destino já existir, o arquivo não é copiado e o fato é registrado.
- **Integridade:**
  - o SHA-256 e o MD5 do **original são recalculados no momento da exportação** e comparados com o índice (se o original mudou desde a indexação, isso é apontado) e com a **cópia**;
  - cada conjunto tem `hashes_sha256.txt` e `hashes_md5.txt` em formato padrão, conferíveis com `certutil -hashfile`, `Get-FileHash` ou `sha256sum -c`;
  - o SHA-256 do manifesto aparece ao final da exportação.
- **O manifesto registra:** ordem, relação, confiança, detalhe, nome, caminho original e relativo, tamanho, datas de criação, modificação e acesso do original, data de captura, dimensões, formato, os três SHA-256 (índice, original agora, cópia), os MD5, as datas da cópia, "datas preservadas" e a conferência. Também registra computador, usuário e horário.
- O programa recusa destino dentro da pasta analisada, e os originais são apenas lidos.

## Como a semelhança é detectada

1. **SHA-256** encontra as cópias idênticas.
2. **pHash** nas 8 orientações encontra miniaturas, caches, versões redimensionadas, recomprimidas, rotacionadas e espelhadas.
3. A **CNN regional** (ResNet-50, 15 regiões e escalas por imagem) faz a triagem de recortes e edições entre centenas de milhares de imagens, por busca vetorial em GPU.
4. **ORB + RANSAC** confirma geometricamente cada candidato, localiza o recorte e classifica a relação.
5. O **mapa de diferenças** mostra as alterações locais depois do alinhamento.

### Conferência de conteúdo (fundos transparentes, margens e layouts iguais)

O pHash compara só a estrutura geral da imagem em baixa resolução. Imagens **diferentes** com o mesmo "molde" chegam a pHash muito próximos: por exemplo, PNG/WebP com **fundo transparente** e personagem centralizado, ou artes de um mesmo site com o mesmo enquadramento. Por isso, todo resultado do pHash é **conferido pelo conteúdo** na miniatura:
- as margens e o fundo (cor da borda) não contam;
- proporções diferentes são rejeitadas;
- se mais de 35% do conteúdo divergir, o resultado é descartado.

Essa conferência vale **só para dizer "é a mesma imagem"**: redimensionada, recomprimida, miniatura, rotacionada ou espelhada. **Recortes não dependem dela.**

Todo par reprovado como "mesma imagem" ainda passa pela verificação geométrica, com pontos ORB e homografia, que não tem restrição de proporção. Por isso o programa reconhece:
- recortes de qualquer proporção;
- **zoom central com a mesma proporção do original** (de 60% a 92% testados);
- corte de bordas;
- recortes ampliados;
- recortes girados.

Pares de mesmo enquadramento que não são cópia limpa também passam pela geometria. Assim, um corte leve de borda é classificado como **recorte** e não como "alterada".

A mesma conferência é aplicada aos candidatos da CNN e aos selos ≈ da Análise global. Nos selos, um vizinho só conta se tiver o mesmo conteúdo ou se for um recorte confirmado por pontos ORB.

### Imagens e regiões uniformes

Imagens sem conteúdo útil para comparação são tratadas à parte: só branco, só preto, cor sólida, ou com mais de 97% dos pixels praticamente iguais. Para elas, o programa só procura **cópias idênticas (SHA-256)**. Elas podem ser listadas pelo filtro "Imagem uniforme".

Nas demais imagens, as regiões uniformes (fundo branco, margens, céu liso) ficam fora da comparação pela CNN. Isso evita casar uma imagem com fundo branco com qualquer imagem toda branca.

A comparação por pHash também exige cores médias compatíveis. A categoria "visualmente semelhante", que se baseia só na CNN, exige ainda pHash próximo.

Índices antigos são atualizados sozinhos na próxima reindexação (F5), em segundos, a partir das miniaturas já salvas, sem reler os arquivos originais.

## Conteúdo do relatório

1. Identificação, com campos em branco para perito, matrícula e procedimento.
2. Metodologia.
3. Para cada imagem selecionada:
   - ficha com caminho, tamanho, formato real, coerência da extensão, dimensões, datas, câmera, GPS, miniatura EXIF, **SHA-256 e MD5**;
   - tabela das semelhantes, com relação, confiança e SHA-256;
   - figuras lado a lado com o recorte destacado e mapa de diferenças (até 15 por imagem).
4. Limitações.
5. Relação de hashes de todos os arquivos citados.

## Integridade do material

- **Somente leitura:** nenhum arquivo da pasta é alterado, e nada é criado dentro dela.
- O índice fica em `%LOCALAPPDATA%\GaleriaPericial\indices\<pasta>_<hash>\indice.db`, um por pasta analisada.
- Espaço do índice: cerca de 15 KB por imagem, ou seja, uns 4-5 GB para 300 mil imagens.
- Para a busca, os vetores ocupam cerca de 1,2 GB por 100 mil imagens na RAM (modo CPU). Na GPU ocupam a metade, divididos entre as placas.
- Os resultados são **indicativos** (triagem) e não substituem o exame pericial.

## Limitações conhecidas

- Recortes muito pequenos (menos de ~5% da área) ou de regiões sem textura podem não ser detectados.
- Os escores da CNN não são probabilidades. O "Limiar ≈" padrão (0,70) é um ponto de partida e deve ser calibrado com casos conhecidos da sua própria coleção.
- Sem a CNN (PyTorch ausente ou falhando), recortes só são procurados por força bruta em coleções de até 5.000 imagens.

## Licença

MIT. Ver o arquivo [LICENSE](LICENSE). Copyright (c) 2026 Artur Lira dos Santos.
