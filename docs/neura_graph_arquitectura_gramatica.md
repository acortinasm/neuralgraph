# **NeuralGraphDB: Arquitectura Técnica y Gramática NGQL**

Versión: 1.0 (Draft)  
Fecha: 2026-01-09  
Destinatario: Equipo de Ingeniería (Rust Core & AI Team)

## **1\. Visión de Arquitectura (Modelo C4)**

### **1.1 Diagrama de Contexto (Nivel 1\)**

Visión general de cómo NeuralGraphDB se integra en el ecosistema de IA.

graph TD  
    User\[Desarrollador de IA / Agente\] \-- "Envía Queries NGQL (gRPC/HTTP)" \--\> NGDB\[NeuralGraphDB System\]  
      
    NGDB \-- "Solicita Embeddings/Completions" \--\> LLM\[External LLM Service\\n(OpenAI/Ollama)\]  
    NGDB \-- "Carga Datos Raw" \--\> S3\[Object Storage\\n(S3/Local Disk)\]  
    NGDB \-- "Zero-Copy Training Data" \--\> PyTorch\[PyTorch / GNN Training\]

    style NGDB fill:\#f9f,stroke:\#333,stroke-width:4px  
    style User fill:\#fff,stroke:\#333  
    style LLM fill:\#e1f5fe,stroke:\#333  
    style PyTorch fill:\#e8f5e9,stroke:\#333

### **1.2 Diagrama de Contenedores (Nivel 2 \- Rust Crates)**

Desglose interno de los módulos del monorepo en Rust. La separación de responsabilidades es crítica para la compilación incremental y el testing.

graph TB  
    subgraph "NeuralGraphDB Server"  
        Server\[neural-server\\n(gRPC Interface)\]  
          
        subgraph "Query Layer"  
            Parser\[neural-parser\\n(Nom / AST)\]  
            Planner\[neural-planner\\n(Logical Optimizer)\]  
        end

        subgraph "Execution Layer"  
            Executor\[neural-executor\\n(Physical Plan)\]  
            GraphBLAS\[neural-math\\n(Sparse Matrix Ops)\]  
            VectorEngine\[neural-vector\\n(HNSW Index)\]  
        end

        subgraph "Storage Layer"  
            Storage\[neural-storage\\n(CSR \+ Arrow)\]  
        end  
    end

    Server \--\> Parser  
    Parser \--\> Planner  
    Planner \--\> Executor  
    Executor \--\> GraphBLAS  
    Executor \--\> VectorEngine  
    GraphBLAS \--\> Storage  
    VectorEngine \--\> Storage

## **2\. Flujo de Ejecución (Sequence Diagram)**

Traza de vida de una consulta híbrida: MATCH (n) WHERE vector(n, "IA") CLUSTER BY leiden RETURN n.

sequenceDiagram  
    participant Client  
    participant Parser  
    participant Planner  
    participant Executor  
    participant MathKernel (GraphBLAS)  
    participant Storage

    Client-\>\>Parser: Envia Query NGQL  
    Note right of Parser: Lexing & Parsing (Nom)  
    Parser-\>\>Planner: Retorna AST (Abstract Syntax Tree)  
      
    Note right of Planner: Optimización  
    Planner-\>\>Planner: Reordenar: Vector Filter PRIMERO, luego Grafo  
    Planner-\>\>Executor: Envia Logical Plan (DAG)

    Note right of Executor: Fase 1: Filtrado Vectorial  
    Executor-\>\>Storage: Scan Index (HNSW)  
    Storage--\>\>Executor: Retorna Bitmask de Nodos Candidatos

    Note right of Executor: Fase 2: Travesía Matricial  
    Executor-\>\>MathKernel: Matrix-Vector Mult (AdjMatrix \* Bitmask)  
    MathKernel--\>\>Executor: Retorna Vector Resultado (Vecinos)

    Note right of Executor: Fase 3: Algoritmo IA  
    Executor-\>\>MathKernel: Ejecutar Leiden(SubGraph)  
    MathKernel--\>\>Executor: Retorna Community IDs

    Executor--\>\>Client: Retorna JSON {nodos, clusters}

## **3\. Especificación Formal de Gramática (EBNF)**

Esta es la "verdad absoluta" para el equipo que desarrolla el Parser. Define la sintaxis válida para el MVP.

(\* EBNF Specification for NGQL v1.0 \*)

(\* Entry Point \*)  
query \= statement , { ";" , statement } ;

statement \=   
    | data\_query\_stmt  
    | load\_stmt  
    ;

(\* Data Query Structure \*)  
data\_query\_stmt \=   
    match\_clause ,   
    \[ where\_clause \] ,   
    \[ cluster\_clause \] , (\* Custom AI Clause \*)  
    \[ generate\_clause \] , (\* Custom AI Clause \*)  
    return\_clause ;

(\* 1\. MATCH Clause \*)  
match\_clause \= "MATCH" , pattern ;  
pattern      \= node\_pattern , { relationship\_pattern , node\_pattern } ;

node\_pattern \= "(" , \[ identifier \] , \[ ":" , label \] , ")" ;  
relationship\_pattern \= "-" , \[ "\[" , \[ ":" , label \] , "\]" \] , "-\>" ;

(\* 2\. WHERE Clause (Hybrid) \*)  
where\_clause \= "WHERE" , expression ;  
expression   \= vector\_expr | logic\_expr ;

vector\_expr  \= "vector\_similarity" , "(" , identifier , "," , ( string\_literal | param ) , ")" , comparator , float ;  
logic\_expr   \= identifier , "." , property , comparator , value ;

(\* 3\. CLUSTER BY (New AI Feature) \*)  
cluster\_clause \= "CLUSTER BY" , algorithm\_call , "INTO" , identifier ;  
algorithm\_call \= algorithm\_name , "(" , \[ params \] , ")" ;  
algorithm\_name \= "leiden" | "pagerank" | "louvain" ;

(\* 4\. GENERATE (New AI Feature) \*)  
generate\_clause \= "GENERATE" , identifier , "USING" , model\_call , "PROMPT" , string\_expression ;  
model\_call      \= "model" , "(" , string\_literal , ")" ;

(\* 5\. RETURN Clause \*)  
return\_clause \= "RETURN" , return\_item , { "," , return\_item } ;  
return\_item   \= identifier \[ "." , property \] | aggregation ;

(\* Basic Tokens \*)  
identifier \= letter , { letter | digit | "\_" } ;  
label      \= identifier ;  
param      \= "$" , identifier ;  
comparator \= "\>" | "\<" | "=" | "\>=" | "\<=" ;

## **4\. Estructura de Proyecto Sugerida (Rust Workspace)**

Para mantener la velocidad de compilación y limpieza de código, usaremos un Cargo Workspace.

/neural-graph-db (Repo Root)  
├── Cargo.toml              \# Workspace definition  
├── /docs                   \# EBNF, Architecture specs  
├── /examples               \# Demo Jupyter notebooks / Python scripts  
├── /crates  
│   ├── /neural-core        \# Tipos base (Node, Edge, Graph traits)  
│   ├── /neural-storage     \# Implementación CSR y Apache Arrow (Parquet)  
│   ├── /neural-math        \# Wrappers de GraphBLAS y algoritmos lineales  
│   ├── /neural-vector      \# Wrapper de HNSW (Faiss o USearch)  
│   ├── /neural-parser      \# Implementación 'nom' de la gramática NGQL  
│   ├── /neural-planner     \# Optimizador de queries (AST \-\> Physical)  
│   └── /neural-server      \# gRPC entry point (Tonic) y Main  
└── /sdk  
    └── /python             \# Cliente Python "pip install neuralgraph"

## **5\. Estrategia de Implementación del Parser (Sprint 1-2)**

1. **Lexer (Tokenizador):** Usar la crate logos para máxima velocidad. Definir enum Token (KwMatch, KwWhere, Ident, etc.).  
2. **Parser (Combinators):** Usar nom.  
   * Crear funciones pequeñas: parse\_node, parse\_rel.  
   * Combinarlas en parse\_pattern.  
   * **Crucial:** Manejar errores de forma amigable. Si falla el parsing de vector\_similarity, el error debe decir "Esperaba un vector, encontré X".  
3. **AST:** Definir structs en neural-core que representen el árbol parseado. Este AST debe ser serializable (serde) para debugging.

## **6\. Notas de Rendimiento (Guidelines)**

* **Zero Allocations:** En el "Hot Path" (ejecución de query), evitar String::clone(). Usar \&str y referencias de vida ('a) siempre que sea posible.  
* **Vectorización:** En neural-math, usar iteradores que el compilador de Rust pueda vectorizar (AVX2/AVX-512).  
* **Async:** neural-server debe ser async (Tokio), pero las operaciones matemáticas intensivas (neural-math) deben correr en un ThreadPool dedicado (Rayon) para no bloquear el Event Loop.