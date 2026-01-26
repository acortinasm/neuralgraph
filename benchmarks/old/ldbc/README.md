# LDBC Social Network Benchmark (SNB) Strategy for NeuralGraphDB

**Target:** SNB Interactive Workload (Scale Factor 0.1 / 1)

## 1. Schema Mapping

The LDBC SNB schema models a realistic social network. We will map the core entities as follows:

### Nodes (Entities)
| LDBC Entity | nGraph Label | Properties |
| :--- | :--- | :--- |
| **Person** | `Person` | `id`, `firstName`, `lastName`, `gender`, `birthday`, `creationDate`, `browserUsed`, `locationIP` |
| **Post** | `Post` | `id`, `imageFile`, `creationDate`, `locationIP`, `browserUsed`, `language`, `content`, `length` |
| **Comment** | `Comment` | `id`, `creationDate`, `locationIP`, `browserUsed`, `content`, `length` |
| **Forum** | `Forum` | `id`, `title`, `creationDate` |
| **Tag** | `Tag` | `id`, `name`, `url` |
| **TagClass** | `TagClass` | `id`, `name`, `url` |
| **Organisation**| `Organisation`| `id`, `type` (University/Company), `name`, `url` |
| **Place** | `Place` | `id`, `name`, `url`, `type` (City/Country/Continent) |

### Edges (Relationships)
| Source | Relationship | Target |
| :--- | :--- | :--- |
| `Person` | `KNOWS` | `Person` |
| `Person` | `LIKES` | `Post` / `Comment` |
| `Person` | `HAS_CREATOR` | `Post` / `Comment` |
| `Person` | `WORK_AT` | `Organisation` |
| `Person` | `STUDY_AT` | `Organisation` |
| `Post` | `HAS_TAG` | `Tag` |
| `Forum` | `HAS_MEMBER` | `Person` |
| `Forum` | `CONTAINER_OF` | `Post` |

## 2. Workload Implementation

We will focus initially on the **Interactive Short Reads** (latency-sensitive lookups) and **Complex Reads** (deep traversals).

### Short Reads (IS)
*   **IS1 (Profile):** Get all properties of a Person.
*   **IS2 (Messages):** Get the last 10 messages created by a Person.
*   **IS3 (Friends):** Get friends of a Person.

### Complex Reads (IC)
*   **IC1 (Friends with name):** Find friends (and friends of friends) with a specific name.
*   **IC2 (Recent messages):** Messages by friends within a date range.

## 3. Execution Plan

1.  **Data Generation:** Download/Generate pre-computed CSVs for Scale Factor 0.1 (approx 100MB) to start.
2.  **Loader:** Python script to clean and bulk-load these CSVs into NeuralGraphDB.
3.  **Runner:** Python script using the `neuralgraph` client to execute the queries and measure latency.
