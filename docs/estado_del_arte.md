Para diseñar una base de datos híbrida (Grafo \+ Vector) con capacidades de IA nativas, los documentos proporcionan una hoja de ruta arquitectónica clara. Las tendencias actuales sugieren abandonar la arquitectura de "doble base de datos" (sincronizar una base de datos vectorial con una de grafos) en favor de un diseño unificado que trate a los vectores como ciudadanos de primera clase dentro del grafo.  
A continuación, presento las características (*features*) y consideraciones arquitectónicas clave que debes incluir, categorizadas según su función en el sistema:

### 1\. Arquitectura de Almacenamiento y Gestión de Datos

El diseño debe permitir que las operaciones vectoriales y de grafos coexistan sin penalizar el rendimiento de ninguna de las dos.

* **Almacenamiento Desacoplado de Vectores:** No almacenes los embeddings simplemente como listas de flotantes junto con otros atributos del nodo. Debes separar físicamente el almacenamiento de los vectores (embedding segments) de los atributos del grafo (vertex segments). Esto permite optimizar la indexación vectorial (usando índices nativos como HNSW) y facilita actualizaciones eficientes y transaccionales sin reconstruir todo el índice del grafo 1, 2\.  
* **Tipo de Dato "Embedding" Nativo:** Implementa un tipo de dato específico para embeddings que gestione metadatos críticos: dimensionalidad, modelo de origen (e.g., GPT-4, CLIP), tipo de índice y métrica de distancia (coseno, euclidiana). Esto facilita la gestión de múltiples espacios vectoriales dentro del mismo grafo y simplifica la carga de datos 3, 4\.  
* **Arquitectura MPP (Massively Parallel Processing):** Para escalar, el sistema debe particionar tanto los vértices del grafo como los segmentos de embeddings, permitiendo búsquedas paralelas y distribuidas. Esto es esencial para mantener baja latencia en grafos de gran escala (miles de millones de nodos) 5, 6\.

### 2\. Capacidades de Búsqueda Híbrida y Avanzada

Tu base de datos debe permitir consultas que mezclen la semántica (vector) con la estructura (grafo) en una sola ejecución.

* **Búsqueda Vectorial Filtrada por Patrones de Grafo:** Una *feature* crítica es poder ejecutar una búsqueda vectorial restringida a un subconjunto de nodos que cumplan un patrón de grafo específico (por ejemplo, "buscar los posts más similares a X, pero solo entre los amigos de Alice"). Esto requiere pre-filtrado eficiente o estrategias de ejecución entrelazada 7, 8\.  
* **Unión de Similitud Vectorial (Vector Similarity Join):** Permite encontrar los pares top-k de nodos (origen, destino) conectados por un patrón de grafo específico que sean semánticamente similares. Esto es útil para detectar entidades duplicadas o analizar evoluciones temporales en relaciones 9\.  
* **Soporte Multimodal con Particionamiento:** Si vas a manejar texto, imágenes y audio, considera el "particionamiento consciente de la modalidad". Esto implica crear índices separados optimizados para cada modalidad (e.g., un índice HNSW para texto y otro para imágenes), lo que reduce el espacio de búsqueda y mejora la precisión al evitar sesgos entre modalidades 10, 11\.

### 3\. Funcionalidades para RAG (Retrieval-Augmented Generation)

Dado que buscas incorporar IA nativa, el sistema debe estar optimizado para flujos de trabajo RAG avanzados, específicamente GraphRAG.

* **Indexación de Doble Nivel (Específico y Abstracto):** Para soportar tanto preguntas puntuales como temáticas amplias, tu base de datos debe indexar la información a nivel de entidades (detalles) y a nivel de comunidades o clústeres (temas abstractos). Esto permite recuperar respuestas coherentes tanto para "¿Quién escribió X?" como para "¿Cómo influye la IA en la educación?" 12, 13\.  
* **Generación de Resúmenes de Comunidad:** Implementa algoritmos de detección de comunidades (como Leiden o Louvain) de forma nativa para agrupar nodos y generar resúmenes jerárquicos. Estos resúmenes deben almacenarse y ser recuperables vectorialmente para responder consultas globales ("Global Search") 14, 15\.  
* **Grafo de Conceptos y Selección de Chunks:** Para optimizar costos y eficiencia, considera implementar un "Grafo de Conceptos" independiente del LLM (basado en co-ocurrencia y similitud). Esto permite seleccionar solo los "Core Chunks" (fragmentos de texto más relevantes) para la construcción costosa del grafo, reduciendo el uso de tokens sin perder precisión 16, 17\.

### 4\. Actualización y Mantenimiento Dinámico

Un problema común en bases de datos vectoriales es la dificultad de actualizar índices en tiempo real.

* **Actualizaciones Incrementales y Atómicas:** El sistema debe soportar la inserción de nuevos documentos y la actualización del grafo de conocimiento sin reconstruir todo el índice. Utiliza mecanismos como MVCC (Multi-Version Concurrency Control) y almacenes delta para gestionar actualizaciones de vectores y grafos de forma transaccional y consistente 18, 19\.  
* **Cuantización Dinámica (Flash Quantization):** Para optimizar el uso de memoria, implementa cuantización (compresión de vectores de 32 bits a 8 bits o menos) que se active dinámicamente según el uso de recursos, manteniendo la precisión de búsqueda 19\.

### 5\. Potencia Expresiva del Grafo (GNNs)

Para que el modelo de grafo sea "provablemente potente" en tareas de IA más allá de la recuperación simple:

* **Adaptaciones GNN Nativas:** Incorpora mecanismos que permitan a las Redes Neuronales de Grafos (GNN) distinguir estructuras complejas. Esto incluye **Paso de Mensajes Inverso** (para capturar flujos de salida), **Numeración de Puertos** (para distinguir entre múltiples transacciones/aristas paralelas) e **Identificadores de Ego** (para romper simetrías y detectar ciclos). Estas adaptaciones son cruciales para tareas como detección de fraude o lavado de dinero 20, 21\.

### Resumen de Componentes Recomendados

Basado en el "Unified Framework" propuesto en los documentos 22, tu base de datos debería exponer estos **Operadores de Recuperación** como primitivas:

1. **Operadores de Nodo:** Búsqueda vectorial (VDB), PageRank Personalizado (PPR) para expansión local.
2. **Operadores de Relación:** Recuperación de aristas basada en similitud semántica o vecinos de 1 salto.
3. **Operadores de Comunidad:** Búsqueda vectorial sobre resúmenes de comunidades pre-calculados.

---

## Estado de Implementación en NeuralGraph

La siguiente tabla compara las características recomendadas con el estado actual de NeuralGraph:

### Leyenda de Estado
- **Implementado**: Funcionalidad completa disponible
- **Parcial**: Implementación básica, requiere extensión
- **Planificado**: En roadmap, infraestructura base existe
- **No considerado**: Sin implementación ni planes actuales

### Leyenda de Prioridad
- **P0 (Crítica)**: Diferenciador clave, necesario para competir
- **P1 (Alta)**: Importante para casos de uso principales
- **P2 (Media)**: Mejora significativa de funcionalidad
- **P3 (Baja)**: Nice-to-have, casos de uso especializados

---

### 1. Arquitectura de Almacenamiento y Gestión de Datos

| Feature | Estado | Prioridad | Notas de Implementación |
|---------|--------|-----------|-------------------------|
| **Almacenamiento Desacoplado de Vectores** | **Implementado** | - | `VectorIndex` (HNSW) + `LSM-VEC` separados de `GraphStore`. Embeddings en segmentos independientes de CSR/CSC. |
| **Tipo de Dato Embedding Nativo** | **Parcial** | P1 | `PropertyValue::Vector(Vec<f32>)` existe. `VectorIndexConfig` tiene dimensión, m, ef. **Falta**: modelo de origen, múltiples métricas de distancia (solo coseno). |
| **Arquitectura MPP** | **Planificado** | P1 | Infraestructura Raft para replicación existe. Arrow Flight para transferencia. **Falta**: Particionamiento de queries distribuidas, sharding de grafos/vectores. |

---

### 2. Capacidades de Búsqueda Híbrida y Avanzada

| Feature | Estado | Prioridad | Notas de Implementación |
|---------|--------|-----------|-------------------------|
| **Búsqueda Vectorial Filtrada por Patrones de Grafo** | **Implementado** | - | `VectorSearch` con filtro de label. `search_filtered()` combina HNSW + predicados. Planner detecta `ORDER BY vector_similarity(...) DESC LIMIT k`. |
| **Unión de Similitud Vectorial (Vector Similarity Join)** | **No considerado** | P2 | Solo k-NN básico implementado. **Falta**: Operador de join para top-k pares (src, dst) con restricciones de patrón. Útil para deduplicación de entidades. |
| **Soporte Multimodal con Particionamiento** | **No considerado** | P3 | Sin particionamiento por modalidad. Actualmente un único índice HNSW. **Futuro**: Índices separados para texto/imagen/audio. |

---

### 3. Funcionalidades para RAG (Retrieval-Augmented Generation)

| Feature | Estado | Prioridad | Notas de Implementación |
|---------|--------|-----------|-------------------------|
| **Indexación de Doble Nivel (Entidad + Comunidad)** | **Parcial** | P0 | Nivel entidad: `LabelIndex`, `PropertyIndex`, `VectorIndex`. **Falta**: Índice vectorial sobre comunidades/clusters para Global Search. |
| **Generación de Resúmenes de Comunidad** | **Parcial** | P0 | `detect_communities_leiden()` implementado (algoritmo Leiden). `Communities` struct almacena asignaciones. **Falta**: Generación automática de resúmenes con LLM e indexación vectorial de los mismos. |
| **Grafo de Conceptos y Selección de Chunks** | **Parcial** | P1 | `EtlPipeline` extrae entidades/relaciones con LLM. PDF loader existe. **Falta**: Selección automática de Core Chunks por centralidad, grafo de co-ocurrencia sin LLM. |

---

### 4. Actualización y Mantenimiento Dinámico

| Feature | Estado | Prioridad | Notas de Implementación |
|---------|--------|-----------|-------------------------|
| **Actualizaciones Incrementales y Atómicas (MVCC)** | **Implementado** | - | `VersionedPropertyStore` con MVCC completo. WAL para durabilidad. `TransactionManager` con snapshot isolation. `vacuum()` para poda de versiones. |
| **Cuantización Dinámica (Flash Quantization)** | **No considerado** | P2 | Vectores almacenados como `f32`. **Falta**: Compresión a int8/binary, activación dinámica según memoria. Reduciría footprint ~4x. |

---

### 5. Potencia Expresiva del Grafo (GNNs)

| Feature | Estado | Prioridad | Notas de Implementación |
|---------|--------|-----------|-------------------------|
| **Paso de Mensajes Inverso** | **Parcial** | P2 | `CscMatrix` permite traversals de aristas entrantes O(1). **Falta**: Operador explícito de message passing inverso para GNNs. |
| **Numeración de Puertos** | **No considerado** | P3 | Sin implementación. Relevante para multi-aristas paralelas (e.g., múltiples transacciones entre mismos nodos). |
| **Identificadores de Ego** | **No considerado** | P3 | Sin implementación. Útil para romper simetrías en detección de fraude/ciclos. |

---

### 6. Operadores de Recuperación (Unified Framework)

| Feature | Estado | Prioridad | Notas de Implementación |
|---------|--------|-----------|-------------------------|
| **Búsqueda Vectorial (VDB)** | **Implementado** | - | `VectorSearch` operator, HNSW index, `vector_similarity()` function. |
| **PageRank Personalizado (PPR)** | **No considerado** | P1 | Sin implementación. Crítico para expansión local en GraphRAG. Alternativa actual: BFS con `ExpandVariableLength`. |
| **Recuperación de Aristas por Similitud Semántica** | **No considerado** | P2 | Solo recuperación por tipo (`EdgeTypeIndex`). **Falta**: Embeddings en aristas + búsqueda vectorial sobre relaciones. |
| **Vecinos de 1 Salto** | **Implementado** | - | `ExpandNeighbors`, `ExpandByType`. O(1) con CSR/CSC. |
| **Búsqueda Vectorial sobre Resúmenes de Comunidad** | **No considerado** | P0 | Detección de comunidades existe, pero sin resúmenes indexados. Bloquea Global Search de GraphRAG. |

---

## Resumen de Prioridades

### P0 - Críticas (Diferenciadores para GraphRAG)
| Feature | Estado | Impacto |
|---------|--------|---------|
| Indexación de Doble Nivel | Parcial | Habilita Local + Global Search |
| Generación de Resúmenes de Comunidad | Parcial | Completa pipeline GraphRAG |
| Búsqueda Vectorial sobre Comunidades | No considerado | Operador core para Global Search |

### P1 - Alta Prioridad
| Feature | Estado | Impacto |
|---------|--------|---------|
| Tipo de Dato Embedding Completo | Parcial | Multi-modelo, multi-métrica |
| Arquitectura MPP | Planificado | Escalabilidad a billones de nodos |
| PageRank Personalizado | No considerado | Expansión local eficiente |
| Selección de Core Chunks | Parcial | Optimización de costos LLM |

### P2 - Media Prioridad
| Feature | Estado | Impacto |
|---------|--------|---------|
| Vector Similarity Join | No considerado | Deduplicación, análisis temporal |
| Cuantización Dinámica | No considerado | Reducción de memoria 4x |
| Paso de Mensajes Inverso | Parcial | Soporte GNN básico |
| Aristas con Embeddings | No considerado | Relaciones semánticas |

### P3 - Baja Prioridad
| Feature | Estado | Impacto |
|---------|--------|---------|
| Soporte Multimodal | No considerado | Casos de uso especializados |
| Numeración de Puertos | No considerado | Multi-aristas (fraude) |
| Identificadores de Ego | No considerado | Detección de ciclos avanzada |

---

## Roadmap Sugerido

**Sprint Actual / Próximo:**
1. Resúmenes de comunidad con LLM + indexación vectorial (P0)
2. Operador de búsqueda sobre comunidades (P0)

**Corto Plazo (1-2 meses):**
3. PageRank Personalizado (P1)
4. Metadatos completos en embeddings (modelo, métrica) (P1)
5. Selección automática de Core Chunks por centralidad (P1)

**Medio Plazo (3-6 meses):**
6. Particionamiento de queries MPP (P1)
7. Vector Similarity Join (P2)
8. Cuantización dinámica int8/binary (P2)

**Largo Plazo:**
9. Soporte multimodal con índices separados (P3)
10. Adaptaciones GNN completas (P3)

