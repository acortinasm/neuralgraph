# **🛡️ NeuralGraphDB v1.0: Roadmap de Ingeniería y Rigor Académico**

Este documento detalla la transición de la arquitectura actual (v0.9) a la **v1.0**, optimizada para el estado del arte en IA y publicación científica.

## **📊 Evaluación de Estado: v0.9 vs v1.0**

| Dimensión | Estado Actual (v0.9) | Requisito v1.0 (Academic Ready) |
| ----- | ----- | ----- |
| **Almacenamiento** | CSR Híbrido (Estático \+ Listas Delta) | **PCSR (Packed Compressed Sparse Row)** |
| **Búsqueda Vectorial** | Índice HNSW como "Sidecar" externo | **Semianillo Neural (HNSW as Matrix)** |
| **Hardware** | Optimización de CPU básica | **SIMD Intrinsics (AVX-512 / NEON)** |
| **GraphRAG** | Travesías Cypher básicas | **Detección de Comunidades (Leiden) Nativa** |

## **🏗️ Épica 1: Sustrato Dinámico (PCSR & PMA)**

**Objetivo:** Implementar la estructura de datos que permite actualizaciones en tiempo real manteniendo la velocidad de escaneo lineal de una matriz compacta.

### **Sprint 1: El Motor PMA (Packed Memory Array)**

* **Análisis de Arquitectura:** Analizar la implementación actual del sistema de archivos y memoria en el prototipo v0.9 para evaluar cómo la introducción del PMA afectará la gestión de punteros y la serialización actual.  
* **User Story 1.1:** Como motor de almacenamiento, quiero gestionar un array con "gaps" (espacios) que se rebalanceen automáticamente para permitir inserciones en $O(\\log^2 N)$ sin reescribir todo el buffer.  
* **Criterios de Aceptación:**  
  * Implementación de lógica de densidad por segmentos (Upper/Lower density thresholds).  
  * Función de `rebalance()` eficiente en Rust.  
* **Output Técnico:** Módulo `neural-storage::pma` probado con 10M de inserciones aleatorias.

### **Sprint 2: Integración PCSR Unificada**

* **Análisis de Arquitectura:** Evaluar el impacto de eliminar las listas de adyacencia dinámicas en el código del `EdgeStore` y cómo la unificación en el PMA alterará los métodos de lectura/escritura concurrentes.  
* **User Story 1.2:** Como desarrollador, quiero eliminar las listas de adyacencia dinámicas y unificar todos los "edges" en el sustrato PMA para evitar *cache misses*.  
* **Criterios de Aceptación:**  
  * Refactorización del `EdgeStore` para usar el offset del PMA.  
  * Benchmark comparativo: Reducción del 40% en latencia de travesía vs v0.9.  
* **Output Técnico:** Integración de `pma` en el core de `neural-core`.

## **🧠 Épica 2: Kernel Neural-Algebraico (GraphBLAS Unified)**

**Objetivo:** Unificar la geometría (vectores) con la topología (grafos) mediante álgebra lineal.

### **Sprint 3: Implementación de Semianillos Neurales**

* **Análisis de Arquitectura:** Analizar el ejecutor de consultas actual (`neural-executor`) para identificar los puntos de integración de los nuevos operadores de semianillo y evaluar la compatibilidad con el sistema de tipos de NGQL.  
* **User Story 2.1:** Como motor de consultas, quiero ejecutar búsquedas vectoriales HNSW como si fueran una multiplicación de matriz dispersa por vector (SpMV) usando un semianillo personalizado.  
* **Criterios de Aceptación:**  
  * Definición de operadores $(\\oplus \= \\text{Top-K}, \\otimes \= \\text{Distancia})$.  
  * Compatibilidad con el ejecutor de queries NGQL.  
* **Output Técnico:** Módulo `neural-algebra::semirings::neural`.

### **Sprint 4: Aceleración SIMD con `faer`**

* **Análisis de Arquitectura:** Evaluar qué kernels algebraicos actuales son críticos para el rendimiento y determinar qué secciones del código de `faer` requieren integración directa con instrucciones AVX-512/NEON.  
* **User Story 2.2:** Como sistema de alto rendimiento, quiero que las operaciones matriciales utilicen instrucciones de hardware específicas (AVX-512) para maximizar el throughput.  
* **Criterios de Aceptación:**  
  * Kernels de `faer` optimizados para la estructura PCSR.  
  * Incremento verificado de 3x en operaciones de agregación.  
* **Output Técnico:** Optimización de `neural-executor` para hardware específico.

## **🕸️ Épica 3: Advanced GraphRAG & Analytics**

**Objetivo:** Proveer las herramientas de análisis global necesarias para agentes de IA modernos.

### **Sprint 5: Algoritmo de Leiden Nativo**

* **Análisis de Arquitectura:** Analizar la arquitectura de concurrencia actual para asegurar que la implementación paralela de Leiden no genere condiciones de carrera en el acceso al PCSR dinámico.  
* **User Story 3.1:** Como analista de IA, quiero ejecutar detección de comunidades jerárquicas (Leiden) directamente en la matriz para agrupar información contextual.  
* **Criterios de Aceptación:**  
  * Implementación paralela del algoritmo sobre el PCSR.  
  * Soporte para grafos pesados (weighted graphs).  
* **Output Técnico:** Módulo `neural-algorithms::community::leiden`.

### **Sprint 6: Pipeline de Resumen de Contexto**

* **Análisis de Arquitectura:** Evaluar el diseño del Parser y el generador de planes de ejecución para incorporar la nueva función `SUMMARIZE` sin romper la compatibilidad con el estándar Cypher actual.  
* **User Story 3.2:** Como agente de IA, quiero que el motor genere automáticamente un "Subgrafo de Conocimiento" (Knowledge Subgraph) basado en una consulta híbrida.  
* **Criterios de Aceptación:**  
  * Función `SUMMARIZE` en NGQL que extrae top-nodes y relaciones clave.  
  * Integración con buffers de contexto para LLMs.  
* **Output Técnico:** Nueva funcionalidad en el Parser y Executor de NGQL.

## **🧪 Épica 4: Validation y Benchmarking (The Paper)**

**Objetivo:** Obtener los datos empíricos para la publicación científica.

### **Sprint 7: Benchmark LDBC y Rigor Comparativo**

* **Análisis de Arquitectura:** Analizar las métricas de instrumentación actuales para asegurar que la captura de datos (latencia, memoria, write amplification) sea precisa y no introduzca un sesgo significativo en los resultados del benchmark.  
* **User Story 4.1:** Como investigador, necesito datos de rendimiento estandarizados contra Neo4j y FalkorDB para validar las tesis del paper.  
* **Criterios de Aceptación:**  
  * Ejecución completa de la suite LDBC Social Network Benchmark.  
  * Gráficas de "Write Amplification" comparando PCSR vs CSR.  
* **Output Técnico:** Dataset de resultados `.csv` y suite de tests de estrés.

## **🛠️ Guía de Evaluación para el Equipo**

Al revisar el código actual, los desarrolladores deben responder:

1. **Concurrencia:** ¿El actual `RwLock` de los indices escala con el nuevo PMA?  
2. **Alineación de Memoria:** ¿Estamos asegurando que los segmentos del PMA estén alineados a líneas de caché (64 bytes)?  
3. **Abstracción de faer:** ¿Estamos usando las APIs de bajo nivel de `faer` o estamos dejando que el compilador autovectorice? (Se prefiere control explícito para el paper).

