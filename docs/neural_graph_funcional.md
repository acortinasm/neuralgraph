# **NeuralGraphDB: Documento de Especificación Funcional**

Versión: 4.1
Fecha: 2026-01-23
Estado: Fase 7 (MVCC Snapshot Isolation) En Progreso - v0.9.2

## **1\. Resumen Ejecutivo**

**NeuralGraphDB** es una base de datos de grafos nativa diseñada específicamente para cargas de trabajo de Inteligencia Artificial (Agentes Autónomos, RAG y GNNs). A diferencia de las bases de datos de grafos tradicionales (basadas en navegación de punteros), NeuralGraphDB utiliza un motor de **álgebra lineal (matrices dispersas)** para unificar la estructura del grafo con representaciones vectoriales.

El objetivo central es eliminar la latencia en el razonamiento de los Agentes de IA y proporcionar capacidades de **GraphRAG** (Retrieval-Augmented Generation) "out-of-the-box", gestionadas por un lenguaje de consulta propio: **NGQL (Neural Graph Query Language)**.

## **2\. Arquitectura Funcional del Sistema**

El sistema se divide en cuatro capas funcionales estrictas para garantizar el rendimiento y la modularidad.

### **2.1 Capa de Interfaz (The Parser)**

* **Componente:** NGQL Custom Parser (basado en logos/Lexer y parser recursivo descendente).  
* **Responsabilidad:** Traducir intenciones de usuario y comandos de IA en un Plan de Ejecución Algebraico. No es un simple traductor SQL; es un compilador de intenciones.

### **2.2 Capa de Inteligencia (The AI Kernels)**

* **Motor Vectorial:** Índice HNSW (Hierarchical Navigable Small World) integrado.  
* **Motor de Algoritmos:** Implementación nativa de algoritmos de grafos (Leiden, PageRank, BFS) ejecutados como operaciones matriciales.  
* **In-Database LLM:** Módulo ligero para inferencia y extracción de entidades (ETL) en la ingesta.

### **2.3 Capa de Cómputo (The Matrix Engine)**

* **Core:** GraphBLAS (Linear Algebra on Graphs) + Streaming Executor.  
* **Lógica:** Todas las operaciones de travesía (MATCH) se convierten en multiplicaciones Matriz-Vector (mxv) o Matriz-Matriz (mxm).  
* **Beneficio:** Paralelización masiva en CPU (SIMD) y preparación para GPU.
* **Modelo de Ejecución:** Streaming (Iteradores perezosos) para manejo eficiente de memoria en grandes grafos.

### **2.4 Capa de Almacenamiento (The Storage)** ✅ IMPLEMENTADO

* **Topología:** Formato CSR (Compressed Sparse Row) para la matriz de adyacencia estática + Listas de Adyacencia Dinámicas para mutaciones.
* **Propiedades:** VersionedPropertyStore con MVCC para snapshot isolation.
* **Persistencia:** Snapshots binarios (`bincode`) + Write-Ahead Log (WAL) para durabilidad.
* **Transacciones:** ACID completo (BEGIN, COMMIT, ROLLBACK) con MVCC para lecturas no bloqueantes.
* **Índices Invertidos (v0.2):**
  * **LabelIndex:** O(1) lookup para `MATCH (n:Label)`
  * **PropertyIndex:** O(1) lookup para `WHERE n.prop = value`
  * **EdgeTypeIndex:** O(1) lookup para `()-[:TYPE]->()`
  * **VectorIndex:** HNSW para búsquedas de similitud (k-NN).

## **3\. Especificación del Lenguaje: NGQL (Neural Graph SQL)**

El producto se diferencia por su lenguaje, que trata los conceptos de IA como ciudadanos de primera clase.

### **3.1 Sintaxis Híbrida**

El parser debe soportar la combinación de tres dominios en una sola consulta:

1. **Patrón (Topología):** MATCH (n)-\[:REL\]-\>(m)  
2. **Vector (Semántica):** WHERE vector\_similarity(n.emb, $query) \> 0.9  
3. **Computación (Algorítmica):** CLUSTER BY / RANK BY / SHORTEST PATH

### **3.2 Comandos Clave (Must-Have)**

#### **CLUSTER BY (Detección de Comunidades)**

Permite agrupar resultados dinámicamente antes de retornarlos, esencial para resumir información en RAG.

MATCH (doc:Document)-\[:MENTIONS\]-\>(topic:Topic)  
WHERE doc.date \> '2025-01-01'  
CLUSTER BY leiden(resolution=1.0) INTO community\_id  
RETURN community\_id, collect(topic.name)

#### **SHORTEST PATH (Algorítmico)**

Búsqueda eficiente de caminos mínimos.

MATCH p = SHORTEST PATH (a)-[*]->(b)
WHERE a.id = 1 AND b.id = 100
RETURN p

#### **WITH GENERATION (Generación RAG)**

Invoca al LLM integrado para sintetizar la respuesta final basada en los datos recuperados.

MATCH (p:Paper) WHERE vector\_search(p.abstract, $question)  
WITH p LIMIT 5  
GENERATE summary USING model('gpt-4o-mini')
PROMPT "Resume estos papers: " \+ p.abstract  
RETURN generated\_summary

#### **LOAD DOCUMENT (Ingesta no estructurada)**

Comando ETL nativo.

LOAD DOCUMENT "s3://bucket/report.pdf"  
CONFIG { "extract\_strategy": "graph\_rag\_triplets" }

## **4\. Módulos Funcionales Detallados**

### **4.1 Módulo de Almacenamiento y Gestión de Datos**

* **Tipos de Datos Soportados:**  
  * Primitivos: String, Integer, Float, Boolean, **Date**, **DateTime**.  
  * Complejos: Array, JSON, **Tensor/Vector** (Array de Floats de longitud fija).  
* **Gestión de Esquema:**  
  * *Schema-Flexible:* Los nodos pueden tener propiedades arbitrarias, pero se optimizan si siguen un esquema definido.  
* **CRUD:** Operaciones atómicas de inserción (`CREATE`), actualización (`SET`) y borrado (`DELETE` con `DETACH`) de nodos/aristas.
* **Persistencia:** Sistema híbrido de snapshots + log de transacciones (WAL).

### **4.2 Módulo de Motor Vectorial (Vector Engine)**

* **Indexación:** Creación automática de índices HNSW al detectar propiedades tipo vector.  
* **Pre-Filtering:** Capacidad de usar filtros de metadatos (ej. date \> 2024\) para restringir el espacio de búsqueda vectorial (Hybrid Search).  
* **Mapping:** Mapeo directo de VectorID a Matrix RowID para evitar "joins" costosos.

### **4.3 Módulo de GraphRAG (Ingesta y Recuperación)**

* **Chunking Inteligente:** División de texto entrante en nodos (:Chunk).  
* **Extracción de Entidades:** Uso de LLM local o API para identificar (:Person), (:Org) y conectarlos automáticamente con (:Chunk).  
* **Resumen Global:** Capacidad de ejecutar algoritmos de centralidad para identificar los nodos más importantes de un subgrafo recuperado.

### **4.4 Interfaz de Memoria Compartida (Zero-Copy)**

* **Apache Arrow Flight Server:** Endpoint dedicado para ciencia de datos.  
* **Funcionalidad:** Permite a clientes externos (Python/PyTorch) solicitar un volcado de memoria de un subgrafo específico sin serialización JSON.  
* **Uso:** Entrenamiento de GNNs (Graph Neural Networks) directamente sobre los datos vivos.

## **5\. Requisitos No Funcionales (SLAs)**

### **5.1 Rendimiento (Latencia)**

* **Consulta Simple (2-hop):** \< 10ms (p95).  
* **Búsqueda Híbrida (Vector \+ 2-hop):** \< 50ms (p95) para grafos de hasta 10M de nodos.  
* **Consultas Agénticas Complejas:** \< 140ms. (Umbral crítico para percepción de tiempo real en voz/chat).

### **5.2 Escalabilidad**

* **Capacidad Vertical:** Soporte eficiente de hasta 100M de nodos y 1B de aristas en una sola máquina (gracias a estructuras dispersas).  
* **Concurrencia:** Soporte para \>1000 consultas de lectura concurrentes por segundo (QPS).
* **Eficiencia de Memoria:** Ejecución streaming para evitar OOM en grandes result sets.

### **5.3 Compatibilidad**

* **Drivers:** SDK oficial de Python (pip install neuralgraph).  
* **Formato de Archivo:** Compatible con lectura directa de archivos Parquet para cargas masivas.

## **6\. Hoja de Ruta de Funcionalidades (Mapping a Fases)**

| Funcionalidad | Fase 1 (Core) | Fase 2 (RAG) | Fase 3 (Infra) | Estado |
| :---- | :---- | :---- | :---- | :---- |
| **Parser NGQL (MATCH, WHERE, RETURN)** | ✅ | ✅ | ✅ | ✅ Completado |
| **Motor CSR (Matrices Dispersas)** | ✅ | ✅ | ✅ | ✅ Completado |
| **Agregaciones (COUNT, SUM, AVG)** | ✅ | ✅ | ✅ | ✅ Completado |
| **ORDER BY, LIMIT, DISTINCT** | ✅ | ✅ | ✅ | ✅ Completado |
| **GROUP BY** | ✅ | ✅ | ✅ | ✅ Completado |
| **Índices (Label, Property, EdgeType)** | ✅ | ✅ | ✅ | ✅ Completado |
| **CLI Interactivo + Docker** | ✅ | ✅ | ✅ | ✅ Completado |
| **Ingesta CSV/Parquet** | ✅ | ✅ | ✅ | ✅ Completado |
| **Índice Vectorial HNSW** | ❌ | ✅ | ✅ | ✅ Completado |
| **Algoritmo Clustering (CLUSTER BY)** | ❌ | ✅ | ✅ | ✅ Completado |
| **Ingesta PDF (LOAD DOCUMENT)** | ❌ | ✅ | ✅ | ✅ Completado |
| **Mutaciones (CREATE, DELETE, SET)** | ❌ | ❌ | ✅ | ✅ Completado |
| **Persistencia (WAL, Snapshots)** | ❌ | ❌ | ✅ | ✅ Completado |
| **Advanced Traversals (Var-Len, Shortest Path)** | ❌ | ❌ | ✅ | ✅ Completado |
| **Explain / Profile** | ❌ | ❌ | ✅ | ✅ Completado |
| **Parameterized Queries ($param)** | ❌ | ❌ | ✅ | ✅ Completado |
| **Streaming Execution** | ❌ | ❌ | ✅ | ✅ Completado |
| **Query Pipelining (WITH, UNWIND)** | ❌ | ❌ | ❌ | ✅ Completado (Fase 4) |
| **Advanced Patterns (OPTIONAL, MERGE)** | ❌ | ❌ | ❌ | ✅ Completado (Fase 4) |
| **Robust Expressions (CASE, String Funcs)** | ❌ | ❌ | ❌ | ✅ Completado (Fase 4) |
| **Temporal Engine (Date, DateTime)** | ❌ | ❌ | ❌ | ✅ Completado (Fase 4) |
| **Interfaz Arrow Flight (Zero-Copy)** | ❌ | ❌ | ❌ | ✅ Completado (Fase 5) |
| **ACID Transactions (BEGIN/COMMIT/ROLLBACK)** | ❌ | ❌ | ❌ | ✅ Completado (Fase 7) |
| **MVCC (Snapshot Isolation)** | ❌ | ❌ | ❌ | ✅ Completado (Fase 7) |
| **Time-Travel (AT TIME)** | ❌ | ❌ | ❌ | Planificado (Fase 7) |

## **7. Estado Actual de Implementación (v0.9 Candidate)**

### Componentes Implementados

| Componente | Crate | Estado |
| :---- | :---- | :---- |
| Tipos Core | `neural-core` | ✅ Completo |
| Parser NGQL | `neural-parser` | ✅ Completo (Soporta Cypher Std, Pipelines, MERGE, Funciones) |
| Almacenamiento | `neural-storage` | ✅ Completo (CSR + Dynamic + WAL + Indices + HNSW + Temporal + MVCC) |
| Executor de Queries | `neural-executor` | ✅ Completo (Streaming + Pushdown + Complex Exprs + MVCC Snapshots) |
| CLI Interactivo | `neural-cli` | ✅ Completo (REPL, Server, Demo, Params) |

### Rendimiento Verificado

| Operación | Target | Actual |
| :---- | :---- | :---- |
| Node lookup | <100ns | 22ns ✅ |
| Label filter | <50ms | <1ms ✅ |
| COUNT(*) | <50ms | <1ms ✅ |
| Vector Search | <50ms | ~10ms (100k nodos) ✅ |
| Streaming | - | O(1) memoria para scans lineales ✅ |

## **8\. Glosario Técnico**

* **CSR (Compressed Sparse Row):** Estructura de datos que comprime una matriz eliminando los ceros, optimizando memoria para grafos.  
* **GraphBLAS:** Estándar de API para primitivas de grafos basadas en álgebra lineal.  
* **Grounding:** Proceso de anclar las respuestas de una IA en datos verificables para evitar alucinaciones.  
* **GNN (Graph Neural Network):** Redes neuronales diseñadas para operar sobre estructuras de grafos.
* **LabelIndex:** Índice invertido que mapea labels a listas ordenadas de NodeIDs para O(1) lookups.
* **PropertyIndex:** Índice invertido que mapea (propiedad, valor) a NodeIDs para filtros WHERE eficientes.
* **EdgeTypeIndex:** Índice que agrupa aristas por tipo para traversals filtrados O(1).
* **WAL (Write-Ahead Log):** Registro secuencial de cambios para garantizar durabilidad ante fallos.
* **Streaming Execution:** Modelo de procesamiento fila a fila (lazy) que evita materializar resultados intermedios en memoria.
* **MVCC (Multi-Version Concurrency Control):** Técnica que mantiene múltiples versiones de datos para permitir lecturas no bloqueantes y snapshot isolation.
* **Snapshot Isolation:** Nivel de aislamiento donde cada transacción ve una instantánea consistente de la base de datos al momento de comenzar.