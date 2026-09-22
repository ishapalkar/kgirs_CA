"""
Virtual Laboratory Experiment: Create and Manage a Graph Database
Developed for Virtual Lab CA / IIT Kharagpur Virtual Labs Style.

A comprehensive, modular laboratory partitioned into 4 core sections:
  1. Theory: Concepts, architecture, LPG model, Cypher reference, setup, procedure, and terminology.
  2. Simulation: Interactive Property Graph Sandbox with visual UI controls, embedded Cypher query engine,
                 real-time Plotly graph visualization, graph metrics, and experimental trial logger.
  3. Quiz: 12-question self-grading conceptual assessment with immediate pedagogical explanations.
  4. Report Generation: Student information, recorded trials, graph snapshot, discussion, and downloadable PDF report.

Note: Streamlit-native components are strictly used to render seamlessly in both light and dark themes.
"""

import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional

warnings.filterwarnings("ignore", message=".*st.components.v1.html.*")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
import json
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Import embedded in-memory graph engine
from graph_engine import PropertyGraph, CypherEngine, Node, Relationship


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Create and Manage a Graph Database",
    "course": "Database Management Systems / Advanced Data Systems",
    "lab_code": "CS-VLAB-08",
    "objectives": [
        "Understand the foundational principles of Graph Databases and the Labeled Property Graph (LPG) model.",
        "Differentiate Graph Databases from Relational Databases (RDBMS) via Index-Free Adjacency (IFA).",
        "Design, build, and manipulate graph structures consisting of nodes, labels, properties, and directed relationships.",
        "Master the Cypher Query Language for pattern matching, creation, updates, and deletions (MATCH, CREATE, SET, DELETE, DETACH DELETE).",
        "Execute single-hop and multi-hop relationship traversals, conditional filtering (WHERE), and aggregations (count, avg).",
        "Understand real-world graph database architecture, deployment procedures (Docker, Cloud, Local), and production use cases."
    ]
}

THEORY_CONTENT = {
    "aim": (
        "To configure a property graph database engine, model connected domain entities "
        "using the Labeled Property Graph (LPG) paradigm, perform CRUD operations using the Cypher query language, "
        "and analyze relationship traversals across interconnected datasets."
    ),
    "learning_objectives": EXPERIMENT_CONFIG["objectives"],
    "introduction": """
### 1. Introduction to Graph Databases
A **Graph Database** is a purpose-built NoSQL database management system designed to store, manage, and query 
highly connected and complex relationship data. Unlike traditional relational database management systems (RDBMS) 
that organize data into rigid tabular rows and columns connected via foreign keys, graph databases treat 
**relationships as first-class citizens** stored directly alongside the entities they connect.

Modern applications—ranging from social media friend graphs, enterprise knowledge graphs, biomedical interaction 
networks, supply-chain logistics, to financial fraud rings—are characterized by dense, deep, and rapidly evolving 
connections. In these domains, traversing relationships in an RDBMS requires expensive, multi-way `JOIN` operations 
that experience exponential performance degradation ($O(N^k)$). Graph databases solve this challenge by adopting 
the **Property Graph Model** combined with **Index-Free Adjacency (IFA)**, enabling constant-time traversal ($O(1)$) 
per hop regardless of the total volume of data stored in the database.
    """,
    "rdbms_vs_graph": r"""
### 2. Graph Database vs. Relational Database (RDBMS)

| Feature / Dimension | Relational Database (RDBMS) | Property Graph Database |
| :--- | :--- | :--- |
| **Primary Data Model** | Tables, Rows (tuples), Columns | Nodes (entities), Directed Relationships (edges) |
| **Relationship Storage** | Foreign keys & associative join tables | Direct physical memory pointers (Index-Free Adjacency) |
| **Deep Join Performance** | Degrades exponentially ($O(N^k)$) with join depth | Constant per-hop traversal time ($O(k)$), independent of total DB size |
| **Schema Flexibility** | Rigid DDL schema; costly schema migrations | Flexible schema / schema-optional; easily evolves |
| **Query Language** | SQL (Structured Query Language) | Declarative Graph Query Languages (Cypher, GQL) |
| **Multi-Hop Traversal** | Recursive Common Table Expressions (CTEs) | Intuitive ASCII-art pattern matching `(a)-[:REL*1..3]->(b)` |
| **Best-Fit Workloads** | Tabular transactions, accounting, structured reporting | Social networks, fraud detection, recommendation engines, knowledge graphs |

#### What is Index-Free Adjacency (IFA)?
In a relational database, traversing a relationship between two tables requires looking up foreign keys using an index 
(typically a $B^+$-Tree with $O(\log N)$ search complexity) or performing hash joins. When traversing multiple hops 
(e.g., *Find friends of friends of Alice*), each hop executes an independent index lookup across millions of records.

In **Native Graph Databases**, each node acts as a direct micro-index to its neighboring nodes. 
A node record holds direct 64-bit physical memory/disk offsets pointing to its connected relationship records, which in 
turn point directly to adjacent nodes. Therefore, traversing an edge requires only dereferencing a memory pointer ($O(1)$). 
The total query execution time is strictly proportional to the **size of the traversed subgraph**, completely independent 
of whether the entire database contains thousands or billions of nodes!
    """,
    "graph_architecture": """
### 3. Graph Database Architecture & Storage Internals
A native property graph database engine manages connected data at the storage, index, and query execution layers:

1. **Storage Layer (Native Graph Storage)**:
   - Graph engines store structures in specialized, fixed-size record files:
     - `nodestore`: Fixed-length records storing node in-use flags, pointer to the first relationship, and pointer to the first property.
     - `relationshipstore`: Fixed-length records containing pointers to source node, target node, relationship type, previous and next relationships for both source and target nodes (doubly-linked relationship chain).
     - `propertystore`: Stores primitive properties (strings, integers, floats, booleans, arrays).
   - Because records are fixed-size, calculating a record's physical file offset is a simple multiplication: $\\text{Offset} = \\text{Record ID} \\times \\text{Record Size}$, enabling instant $O(1)$ random disk access.

2. **Execution Engine (Cypher Runtime)**:
   - Cypher queries are parsed into Abstract Syntax Trees (AST), validated, optimized by a cost-based query planner, and compiled into an executable pipeline.
   - Graph engines utilize iterator models and pipelined runtimes to stream results with minimal memory overhead.

3. **Core Property Graph Elements**:
   - **Nodes**: Discrete domain entities (e.g., a student `Alice`, a course `DBMS`, a department `CSE`). Nodes can possess zero, one, or multiple labels.
   - **Labels**: Tags used to categorize nodes into semantic groups or roles (e.g., `:Student`, `:Faculty`, `:Course`). Labels serve as entry-point indexes.
   - **Properties**: Arbitrary key-value pairs associated with either nodes or relationships (e.g., `{name: 'Alice', gpa: 9.15}`).
   - **Relationships**: Directed connections between two nodes. Every relationship **must have a name/type** (e.g., `[:ENROLLED_IN]`), a designated start node, and an end node. Relationships can also hold properties (e.g., `{semester: '5th', grade: 'Ex'}`).
   - **Directionality**: While relationships are stored with a definite direction (from start node to end node), Cypher queries can traverse them in outgoing `(a)-[r]->(b)`, incoming `(a)<-[r]-(b)`, or bidirectional / undirected `(a)-[r]-(b)` modes.
    """,
    "cypher_crud": """
### 4. Cypher Query Language & CRUD Operations
Cypher is a declarative graph query language that utilizes visual, **ASCII-art syntax** to represent graph patterns.

#### Basic Pattern Grammar:
- **Nodes** are surrounded by parentheses: `(n)`, `(s:Student)`, `(s:Student {name: 'Alice'})`
- **Relationships** are enclosed in brackets with arrows: `-[r:ENROLLED_IN]->`, `<-[:TEACHES]-`, `-[r]-`
- **Paths** combine nodes and relationships: `(s:Student)-[:ENROLLED_IN]->(c:Course)`

#### Cypher CRUD Commands:
1. **CREATE (Create entities and connections)**:
   ```cypher
   // Create a new Student node
   CREATE (s:Student {id: 's_rahul', name: 'Rahul Sen', dept: 'CSE', gpa: 9.20})
   RETURN s;

   // Create a relationship between existing nodes
   MATCH (s:Student {id: 's_rahul'}), (c:Course {code: 'CS101'})
   CREATE (s)-[r:ENROLLED_IN {semester: '1st', grade: 'A'}]->(c)
   RETURN s, r, c;
   ```

2. **MATCH & RETURN (Read and pattern search)**:
   ```cypher
   // Find all courses a student is enrolled in
   MATCH (s:Student {name: 'Alice Smith'})-[r:ENROLLED_IN]->(c:Course)
   RETURN s.name, c.name, r.grade;
   ```

3. **WHERE (Filtering)**:
   ```cypher
   // Filter by numeric threshold and string pattern
   MATCH (s:Student)-[:ENROLLED_IN]->(c:Course)
   WHERE s.gpa >= 8.5 AND c.credits >= 4
   RETURN s.name, s.gpa, c.name, c.credits;
   ```

4. **SET & REMOVE (Update properties and labels)**:
   ```cypher
   // Update property value
   MATCH (s:Student {id: 's_rahul'})
   SET s.gpa = 9.40, s.status = 'Dean List'
   RETURN s;
   ```

5. **DELETE vs. DETACH DELETE (Delete nodes and relationships)**:
   - `DELETE n`: Deletes node `n`. **Constraint:** If node `n` has any attached relationships, the database engine raises an integrity violation exception to preserve graph referential integrity.
   - `DETACH DELETE n`: Automatically deletes all incoming and outgoing relationships connected to `n`, then deletes the node itself.
   ```cypher
   // Delete relationship only
   MATCH (s:Student {id: 's_rahul'})-[r:ENROLLED_IN]->(c:Course {code: 'CS101'})
   DELETE r;

   // Safe deletion of node with relationships
   MATCH (s:Student {id: 's_rahul'})
   DETACH DELETE s;
   ```

6. **Aggregation & Grouping**:
   ```cypher
   // Count enrolled students per course (implicit GROUP BY)
   MATCH (c:Course)<-[:ENROLLED_IN]-(s:Student)
   RETURN c.name, count(s) AS total_students, avg(s.gpa) AS avg_gpa;
   ```

7. **Multi-Hop Traversal (Path Finding)**:
   ```cypher
   // Two-hop pattern: Find faculty who teach courses attended by Alice
   MATCH (s:Student {name: 'Alice Smith'})-[:ENROLLED_IN]->(c:Course)<-[:TEACHES]-(f:Faculty)
   RETURN s.name, c.name, f.name;

   // Variable-length prerequisite chain (1 to 3 hops)
   MATCH (c1:Course)-[:PREREQUISITE_OF*1..3]->(c2:Course)
   RETURN c1.name, c2.name;
   ```
    """,
    "setup_procedure": """
### 5. Prerequisites & Environment Setup

#### Deployment Options:

* **Option A: Containerized Deployment (Recommended)**
  Execute the following command in PowerShell or Terminal to spin up an isolated graph instance:
  ```bash
  docker run -d \\
    --name graph-lab \\
    -p 7474:7474 -p 7687:7687 \\
    -e AUTH_ENABLED=true \\
    -v graph_data:/data \\
    graph-engine:latest
  ```
  - Port `7474`: HTTP browser interface and visual explorer.
  - Port `7687`: Bolt binary protocol for high-performance driver communication.

* **Option B: Managed Cloud Instance**
  1. Create a free managed cloud graph database instance.
  2. Save your connection URI (`bolt://...` or `graph+s://...`).
  3. Query directly via web workspace or driver connection.

* **Option C: Command-Line Interface (Cypher Shell)**
  Launch the CLI client to execute interactive queries:
  ```bash
  cypher-shell -u db_user -p secret_pass
  ```
    """,
    "procedure": [
        "Step 1: Review the theoretical framework, LPG model, and Cypher syntax conventions.",
        "Step 2: Navigate to the 'Simulation' section from the left navigation menu.",
        "Step 3: Select and load a domain preset graph (e.g., University Academic Graph) or start with a blank graph.",
        "Step 4: Use the Visual Graph Builder tabs to create at least one new node (e.g., Student or Course) and one directed relationship.",
        "Step 5: Switch to the Cypher Query Editor tab. Load and execute pre-built example queries (MATCH, WHERE, aggregations).",
        "Step 6: Write custom Cypher queries to perform property updates (SET) and deletion (DETACH DELETE).",
        "Step 7: Inspect the real-time Plotly graph visualization and observe matched node highlighting.",
        "Step 8: Click 'Record Current State / Action' in the Data Log Book to log experimental trials.",
        "Step 9: Complete the 12-question Concept Assessment Quiz to test your graph database mastery.",
        "Step 10: Open 'Report Generation', enter your student credentials and analytical observations, and export your official PDF lab report."
    ],
    "precautions": """
### 6. Precautions & Common Mistakes in Graph Databases

1. **Attempting DELETE on Connected Nodes**:
   - Running `DELETE n` on a node with existing relationships triggers a referential constraint violation. Always use `DETACH DELETE n` if you intend to remove the node along with its connections.
2. **Unintended Cartesian Products in MATCH**:
   - Writing disconnected MATCH patterns like `MATCH (a:Student), (b:Course) RETURN a, b` computes an exhaustive Cartesian product of every student paired with every course ($O(V_1 \\times V_2)$). Always specify relationship patterns between entities: `MATCH (a)-[:ENROLLED_IN]->(b)`.
3. **Case Sensitivity in Cypher**:
   - Cypher keywords (`MATCH`, `WHERE`, `RETURN`) are case-insensitive, but **node labels** (`:Student`), **relationship types** (`[:ENROLLED_IN]`), and **property keys** (`name`, `gpa`) are strictly case-sensitive!
4. **Neglecting Relationship Directionality**:
   - Queries with directed arrows `(a)-[:REL]->(b)` will only return paths matching that exact traversal direction. Use undirected patterns `(a)-[:REL]-(b)` when bidirectional traversals are desired.
5. **Over-Indexing vs. Traversal**:
   - In graph databases, indexes should be created primarily on lookup properties (like student ID or email) to find initial starting nodes. Do not index relationships; graph traversal pointers (IFA) already provide instant traversal.
    """,
    "key_terms": {
        "Node (Vertex)": "A fundamental graph entity representing a distinct object (e.g., Student, Faculty, Course).",
        "Label": "A semantic tag applied to nodes for categorization, schema definition, and indexing (e.g., :Student, :Faculty, :Course).",
        "Relationship (Edge)": "A directed connection between two nodes with a mandatory type and direction (e.g., [:ENROLLED_IN]).",
        "Property": "A key-value attribute associated with a node or relationship (e.g., gpa: 9.15, credits: 4).",
        "Index-Free Adjacency (IFA)": "Architecture where nodes hold direct physical memory pointers to adjacent relationships and nodes, ensuring O(1) traversal.",
        "Cypher": "The declarative, ASCII-art pattern matching query language standardized under openCypher and ISO GQL.",
        "DETACH DELETE": "A Cypher operation that safely deletes a node by first stripping all connected incoming and outgoing relationships.",
        "Degree of a Node": "The total number of relationships connected to a node (Degree = In-Degree + Out-Degree).",
        "Graph Density": "Ratio of existing relationships to the maximum possible directed relationships between nodes: D = E / (V * (V - 1)).",
        "Path": "A continuous alternating sequence of nodes and relationships connecting a start node to an end node.",
        "Multi-Hop Traversal": "A query path that navigates across two or more consecutive relationships (e.g., (a)-[]->(b)-[]->(c)).",
        "Isolated Node": "A node having a degree of zero (no incoming and no outgoing relationships)."
    }
}

# ======================================================================================
# 2. CONCEPT ASSESSMENT QUIZ (12 RIGOROUS QUESTIONS)
# ======================================================================================

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What core architectural feature enables graph databases to achieve constant-time O(1) traversal per hop, unlike RDBMS multi-table joins?",
        "options": [
            "A) Distributed B-Tree indexes on foreign keys",
            "B) Index-Free Adjacency (IFA) using direct physical memory pointers",
            "C) Precomputed materialized relational views",
            "D) Columnar compressed storage files"
        ],
        "answer_index": 1,
        "explanation": "Index-Free Adjacency (IFA) means every node stores direct physical memory pointers to its adjacent relationships, allowing traversal in O(1) time without index searches."
    },
    {
        "id": 2,
        "question": "In the Labeled Property Graph (LPG) model, which of the following statements regarding relationships is FALSE?",
        "options": [
            "A) Every relationship must have a start node and an end node",
            "B) Every relationship must have a specific type (e.g., [:ENROLLED_IN])",
            "C) Relationships can hold key-value properties just like nodes",
            "D) Relationships can exist as dangling pointers without a target node"
        ],
        "answer_index": 3,
        "explanation": "Relationships in a Property Graph are strictly first-class directed connections. They can never exist as dangling pointers without both a valid source and target node."
    },
    {
        "id": 3,
        "question": "Which Cypher pattern correctly matches a Student named 'Alice' who is enrolled in any Course?",
        "options": [
            "A) SELECT Student WHERE name='Alice' JOIN Course",
            "B) MATCH (s:Student {name: 'Alice'})-[:ENROLLED_IN]->(c:Course) RETURN s, c",
            "C) FIND (s:Student)-[ENROLLED_IN]->(c:Course) FILTER s.name = 'Alice'",
            "D) MATCH {s:Student} --> {c:Course} WHERE name = 'Alice'"
        ],
        "answer_index": 1,
        "explanation": "Cypher uses ASCII-art syntax: nodes are enclosed in parentheses '(s:Student)' and directed relationships in bracketed arrows '-[:ENROLLED_IN]->'."
    },
    {
        "id": 4,
        "question": "What happens if you execute 'MATCH (n:Student {id: 'S01'}) DELETE n' when node 'S01' currently has 3 active relationships?",
        "options": [
            "A) The node and its 3 relationships are automatically deleted without error",
            "B) The 3 relationships are preserved as dangling pointers with null sources",
            "C) The graph DBMS engine prevents deletion to preserve referential integrity",
            "D) The node is deleted and the target nodes are also recursively deleted"
        ],
        "answer_index": 2,
        "explanation": "To preserve graph referential integrity, plain DELETE fails if relationships are attached. 'DETACH DELETE' must be explicitly used to remove attached relationships first."
    },
    {
        "id": 5,
        "question": "What is the primary operational role of 'Labels' attached to nodes in a Property Graph?",
        "options": [
            "A) To store arbitrary floating-point numeric measurements",
            "B) To categorize nodes into domain groups and act as entry-point indexes for fast query lookup",
            "C) To define foreign key cascade rules between tables",
            "D) Labels are purely cosmetic and have no execution impact"
        ],
        "answer_index": 1,
        "explanation": "Labels group nodes into semantic roles (e.g., :Student, :Faculty) and allow graph engines to index and rapidly locate starting nodes for graph traversals."
    },
    {
        "id": 6,
        "question": "Which Cypher statement correctly updates the GPA of student 'Alice' to 9.50 and adds an 'honors' property?",
        "options": [
            "A) UPDATE (s:Student {name: 'Alice'}) SET gpa = 9.50, honors = true",
            "B) MATCH (s:Student {name: 'Alice'}) SET s.gpa = 9.50, s.honors = true RETURN s",
            "C) MODIFY Student Alice (gpa: 9.50, honors: true)",
            "D) ALTER NODE (s:Student) WHERE name='Alice' ADD gpa=9.50"
        ],
        "answer_index": 1,
        "explanation": "In Cypher, property updates are performed using the 'SET' clause following a 'MATCH' pattern: 'MATCH (s) SET s.prop = val'."
    },
    {
        "id": 7,
        "question": "In Cypher, what is the behavior of the aggregation function 'count(s)' in 'MATCH (c:Course)<-[:ENROLLED_IN]-(s:Student) RETURN c.name, count(s)'?",
        "options": [
            "A) It throws a syntax error because Cypher requires an explicit 'GROUP BY' clause",
            "B) It automatically groups by non-aggregated fields (c.name) and counts students per course",
            "C) It counts all students in the database regardless of course",
            "D) It only counts courses, ignoring students completely"
        ],
        "answer_index": 1,
        "explanation": "Unlike SQL, Cypher has implicit grouping: any non-aggregated expressions in the RETURN clause (such as c.name) automatically serve as grouping keys."
    },
    {
        "id": 8,
        "question": "What does the variable-length Cypher relationship pattern '-[:PREREQUISITE_OF*1..3]->' express?",
        "options": [
            "A) A relationship whose weight is between 1.0 and 3.0",
            "B) A path of between 1 and 3 sequential PREREQUISITE_OF hops between entities",
            "C) A relationship that must be traversed exactly 3 times in a loop",
            "D) An array of 3 distinct relationship property keys"
        ],
        "answer_index": 1,
        "explanation": "Syntax '*minHops..maxHops' specifies variable-length path traversal. '*1..3' searches for paths having from 1 up to 3 consecutive relationship hops."
    },
    {
        "id": 9,
        "question": "In which scenario would a Relational Database (RDBMS) typically outperform a Graph Database?",
        "options": [
            "A) Finding mutual friends across 6 degrees of separation in a social network",
            "B) Detecting circular money laundering rings across transaction accounts",
            "C) Sequential bulk aggregations across millions of independent, flat accounting records",
            "D) Finding shortest paths through an international airline flight network"
        ],
        "answer_index": 2,
        "explanation": "Relational databases and columnar engines excel at bulk, linear scans and aggregations across flat tables with minimal inter-record joins, whereas graph databases excel at complex, multi-hop relationship traversals."
    },
    {
        "id": 10,
        "question": "Why is the query 'MATCH (s:Student), (c:Course) RETURN s, c' generally discouraged unless explicitly intended?",
        "options": [
            "A) Because it causes a syntax error in Cypher",
            "B) Because it produces an unconstrained Cartesian product matching every student with every course",
            "C) Because it automatically deletes all students and courses",
            "D) Because it forces the database to convert into an RDBMS table"
        ],
        "answer_index": 1,
        "explanation": "Matching disconnected entities without relationship patterns calculates a full Cartesian product (O(|V1| * |V2|)), which can consume massive memory on large graphs."
    },
    {
        "id": 11,
        "question": "Which of the following describes the default network port used for Bolt binary protocol driver communication in standard graph DBMS servers?",
        "options": [
            "A) Port 7474 (HTTP Browser interface)",
            "B) Port 7687 (Bolt binary protocol)",
            "C) Port 3306 (MySQL default port)",
            "D) Port 5432 (PostgreSQL default port)"
        ],
        "answer_index": 1,
        "explanation": "Graph DBMS servers typically use port 7474 for HTTP web console access, and port 7687 for high-performance Bolt binary protocol connections utilized by official drivers."
    },
    {
        "id": 12,
        "question": "According to Cypher naming conventions and syntax standards, how should relationship types and node labels be cased?",
        "options": [
            "A) Labels in UPPER_CASE and Relationships in lower_case",
            "B) Labels in UpperCamelCase (e.g., :Student) and Relationships in UPPER_SNAKE_CASE (e.g., [:ENROLLED_IN])",
            "C) Both labels and relationships must always be lowercase",
            "D) Cypher forbids the use of underscores in relationship types"
        ],
        "answer_index": 1,
        "explanation": "Cypher naming conventions dictate UpperCamelCase for Node Labels (e.g. :Student, :Course) and UPPER_SNAKE_CASE for Relationship Types (e.g. [:ENROLLED_IN], [:TEACHES])."
    }
]


# ======================================================================================
# 3. LAB REPORT PDF EXPORTER
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 130, 140)
        self.cell(0, 10, f"Page {self.page_no()} | Virtual Laboratory CA - Property Graph Database Experiment", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str, graph_metrics: Dict[str, Any]) -> bytes:
    """Compiles experiment benchmark records into an official, publication-quality PDF report."""
    pdf = LabReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Document Header
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, EXPERIMENT_CONFIG["title"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 6, f"{EXPERIMENT_CONFIG['course']} | Course Code: {EXPERIMENT_CONFIG['lab_code']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 27, 190, 22, "FD")

    pdf.set_xy(14, 29)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, student_name or "N/A", new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Roll / ID Number:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, student_id or "N/A", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(14, 38)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, date_str or datetime.now().strftime("%Y-%m-%d"), new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "B", 9)
    pct = int((quiz_score / quiz_total) * 100 if quiz_total else 0)
    if pct >= 50:
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({pct}%)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(12)

    # 1. Learning Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Experiment Objectives & Aim", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 4.5, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(0, 4.5, f" {clean_obj}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # 2. Final Graph Snapshot & Metrics
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Final Graph State & Metric Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)

    summary_text = (
        f"Total Nodes: {graph_metrics.get('num_nodes', 0)}   |   "
        f"Total Relationships: {graph_metrics.get('num_relationships', 0)}   |   "
        f"Labels: {', '.join(graph_metrics.get('labels', [])) or 'None'}\n"
        f"Rel Types: {', '.join(graph_metrics.get('rel_types', [])) or 'None'}   |   "
        f"Graph Density: {graph_metrics.get('density', 0.0)}   |   "
        f"Average Degree: {graph_metrics.get('avg_degree', 0.0)}"
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(4)

    # 3. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Recorded Experimental Trials & Execution Log", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No experimental simulation trials recorded during this session.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)

        # Fixed column widths matching 190 mm
        col_widths = {
            "Trial #": 14,
            "Operation": 28,
            "Query / Action": 68,
            "Result": 42,
            "Status": 18,
            "Timestamp": 20
        }

        # Header
        for c in trials_df.columns:
            w = col_widths.get(c, 25)
            pdf.cell(w, 6, str(c)[:16], border=1, align="C", fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()

        # Rows
        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7.5)
        fill = False

        for _, row in trials_df.iterrows():
            for c in trials_df.columns:
                w = col_widths.get(c, 25)
                val_str = str(row[c])
                pdf.cell(w, 5, val_str[:38], border=1, align="C", fill=fill, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln()
            fill = not fill

    pdf.ln(5)

    # 4. Student Discussion & Observations
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "4. Student Observations & Analytical Discussion", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The graph database experiment successfully demonstrated node and relationship creation, "
        "Cypher pattern matching, property updates, and referential constraints (DETACH DELETE). "
        "Graph traversal queries executed with high efficiency without requiring relational table joins."
    )
    pdf.multi_cell(0, 5, notes_text)
    pdf.ln(8)

    # Sign-off line
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())


def generate_pdf_certificate(student_name: str, student_id: str, institution: str,
                             date_str: str, quiz_score: int, quiz_total: int, cert_id: str) -> bytes:
    """Generates an official landscape certificate of completion as PDF bytes."""
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # Outer border (Deep Navy)
    pdf.set_draw_color(30, 58, 138)
    pdf.set_line_width(2.0)
    pdf.rect(10, 10, 277, 190)

    # Inner ornamental border (Warm Gold)
    pdf.set_draw_color(217, 119, 6)
    pdf.set_line_width(0.8)
    pdf.rect(14, 14, 269, 182)

    # Corner ornamental lines
    pdf.set_draw_color(30, 58, 138)
    pdf.set_line_width(0.5)
    pdf.line(14, 22, 22, 14)
    pdf.line(283, 22, 275, 14)
    pdf.line(14, 188, 22, 196)
    pdf.line(283, 188, 275, 196)

    # Top Header
    pdf.set_xy(20, 22)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(257, 5, "VIRTUAL LABORATORY  |  MINISTRY OF EDUCATION INITIATIVE", align="C")

    pdf.set_xy(20, 27)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(257, 7, (institution or "Department of Computer Science & Engineering").upper(), align="C")

    # Title
    pdf.set_xy(20, 42)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(257, 12, "CERTIFICATE OF COMPLETION", align="C")

    pdf.set_xy(20, 56)
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(257, 7, "This is to certify that", align="C")

    # Student Name
    pdf.set_xy(20, 68)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(2, 132, 199)
    pdf.cell(257, 10, student_name or "Student Participant", align="C")

    # Student ID
    pdf.set_xy(20, 80)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(257, 6, f"Roll / Registration No: {student_id or 'N/A'}", align="C")

    # Body Description
    pdf.set_xy(35, 91)
    pdf.set_font("Helvetica", "", 11.5)
    pdf.set_text_color(51, 65, 85)
    body_text = (
        "has successfully conducted the practical laboratory session, completed the required graph modeling "
        "and Cypher query execution tasks, and demonstrated competence in the foundational practical experiment:"
    )
    pdf.multi_cell(227, 6, body_text, align="C")

    # Experiment Title
    pdf.set_xy(20, 108)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(257, 8, EXPERIMENT_CONFIG["title"].upper(), align="C")

    pdf.set_xy(20, 116)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(257, 6, f"{EXPERIMENT_CONFIG['course']} | Lab Code: {EXPERIMENT_CONFIG['lab_code']}", align="C")

    # Performance summary box
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(60, 128, 177, 18, "FD")

    pdf.set_xy(65, 131)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(55, 6, f"Assessment: {quiz_score} / {quiz_total} Score", align="L")
    pdf.cell(60, 6, f"Date: {date_str}", align="C")
    pdf.cell(50, 6, "Status: Verified Completed", align="R")

    pdf.set_xy(65, 138)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(165, 5, f"Credential Verification ID: {cert_id}", align="C")

    # Signatures
    pdf.set_xy(40, 158)
    pdf.set_draw_color(148, 163, 184)
    pdf.line(40, 173, 100, 173)
    pdf.set_xy(40, 175)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, "Course Coordinator", align="C")
    pdf.set_xy(40, 180)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(60, 4, "Virtual Laboratories Network", align="C")

    # Official Seal in Center
    pdf.set_xy(125, 154)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(180, 83, 9)
    pdf.cell(47, 5, "[ OFFICIAL SEAL ]", align="C")
    pdf.set_xy(125, 160)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(47, 4, "VERIFIED ACADEMIC LAB", align="C")
    pdf.set_xy(125, 165)
    pdf.set_font("Helvetica", "", 7)
    pdf.cell(47, 4, "DIGITAL CERTIFICATE", align="C")

    pdf.line(197, 173, 257, 173)
    pdf.set_xy(197, 175)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, "System Administrator", align="C")
    pdf.set_xy(197, 180)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(60, 4, "Evaluation & Accreditation Unit", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 4. GRAPH PRESETS, SCHEMAS & VISUALIZERS
# ======================================================================================

PRESET_SCHEMAS = {
    "University Academic Knowledge Graph (Default)": {
        "node_labels": ["Student", "Course", "Faculty", "Department", "Project", "Custom..."],
        "default_properties": {
            "Student": {"id_prefix": "s_", "name": "Kiran Patel", "key1": "dept", "val1": "CSE", "key2": "gpa", "val2": "8.80"},
            "Course": {"id_prefix": "cs_", "name": "Machine Learning", "key1": "code", "val1": "CS401", "key2": "credits", "val2": "4"},
            "Faculty": {"id_prefix": "prof_", "name": "Prof. V. Rao", "key1": "dept", "val1": "CSE", "key2": "role", "val2": "Professor"},
            "Department": {"id_prefix": "dept_", "name": "Information Technology", "key1": "code", "val1": "IT", "key2": "building", "val2": "Aryabhatta"},
            "Project": {"id_prefix": "proj_", "name": "Graph Neural Networks", "key1": "domain", "val1": "AI", "key2": "budget", "val2": "150000"},
            "Custom...": {"id_prefix": "custom_", "name": "Custom Entity", "key1": "type", "val1": "General", "key2": "status", "val2": "Active"},
        },
        "rel_types": ["ENROLLED_IN", "TEACHES", "PREREQUISITE_OF", "BELONGS_TO", "ADVISES", "OFFERED_BY", "Custom..."],
        "default_rel_properties": {
            "ENROLLED_IN": {"key": "grade", "val": "A"},
            "TEACHES": {"key": "academic_year", "val": "2024-25"},
            "PREREQUISITE_OF": {"key": "mandatory", "val": "True"},
            "BELONGS_TO": {"key": "since", "val": "2018"},
            "ADVISES": {"key": "project", "val": "Graph Optimization"},
            "OFFERED_BY": {"key": "semester", "val": "Autumn"},
            "Custom...": {"key": "weight", "val": "1.0"},
        },
        "presentation_example": {
            "title": "Add ResearchLab Entity (University Academic Graph)",
            "scenario": "Demonstrate adding a new Research Laboratory entity and linking a Faculty member as director.",
            "label": "ResearchLab",
            "id": "lab_nlp",
            "name": "NLP & AI Research Lab",
            "key1": "director",
            "val1": "Prof. S. Banerjee",
            "key2": "grants_inr",
            "val2": "5000000",
            "followup": "Connect prof_banerjee -[:DIRECTS]-> lab_nlp"
        },
        "cypher_examples": {
            "-- Select an Educational Cypher Example --": "",
            "1. Match all nodes and inspect entire graph": "MATCH (n) RETURN n",
            "2. Match enrolled students and their courses": "MATCH (s:Student)-[:ENROLLED_IN]->(c:Course) RETURN s.name, c.name, s.gpa",
            "3. Filter high-performing students (WHERE clause)": "MATCH (s:Student) WHERE s.gpa >= 8.5 RETURN s.name, s.dept, s.gpa",
            "4. Multi-hop traversal: Faculty teaching enrolled students": "MATCH (f:Faculty)-[:TEACHES]->(c:Course)<-[:ENROLLED_IN]-(s:Student) RETURN f.name, c.name, s.name",
            "5. Aggregate enrollment counts per course (count)": "MATCH (c:Course)<-[:ENROLLED_IN]-(s:Student) RETURN c.name, count(s) AS total_enrolled",
            "6. Prerequisite chain traversal (Course -> Course)": "MATCH (c1:Course)-[:PREREQUISITE_OF]->(c2:Course) RETURN c1.name, c2.name",
            "7. CREATE a new Student node": "CREATE (s:Student {id: 's_kiran', name: 'Kiran Patel', dept: 'CSE', gpa: 8.75}) RETURN s",
            "8. Connect new Student to Course (CREATE relationship)": "MATCH (s:Student {id: 's_kiran'}), (c:Course {name: 'Database Management Systems'}) CREATE (s)-[:ENROLLED_IN {grade: 'A', semester: '5th'}]->(c) RETURN s, c",
            "9. Update student GPA using SET": "MATCH (s:Student {id: 's_kiran'}) SET s.gpa = 9.40 RETURN s",
            "10. Safely remove student using DETACH DELETE": "MATCH (s:Student {id: 's_kiran'}) DETACH DELETE s"
        }
    },
    "Social Network & Friendships": {
        "node_labels": ["Person", "Group", "Topic", "Event", "Custom..."],
        "default_properties": {
            "Person": {"id_prefix": "p_", "name": "Elena Rostova", "key1": "city", "val1": "Mumbai", "key2": "age", "val2": "25"},
            "Group": {"id_prefix": "g_", "name": "Deep Learning Club", "key1": "members_count", "val1": "220", "key2": "category", "val2": "Technology"},
            "Topic": {"id_prefix": "i_", "name": "Knowledge Graphs", "key1": "domain", "val1": "Databases", "key2": "level", "val2": "Advanced"},
            "Event": {"id_prefix": "e_", "name": "Graph Summit 2024", "key1": "location", "val1": "Virtual", "key2": "attendees", "val2": "500"},
            "Custom...": {"id_prefix": "custom_", "name": "Custom Entity", "key1": "category", "val1": "Social", "key2": "active", "val2": "True"},
        },
        "rel_types": ["FRIENDS_WITH", "MEMBER_OF", "INTERESTED_IN", "FOLLOWS", "ORGANIZED", "Custom..."],
        "default_rel_properties": {
            "FRIENDS_WITH": {"key": "since", "val": "2023"},
            "MEMBER_OF": {"key": "role", "val": "Moderator"},
            "INTERESTED_IN": {"key": "level", "val": "Expert"},
            "FOLLOWS": {"key": "since", "val": "2024"},
            "ORGANIZED": {"key": "role", "val": "Lead Organizer"},
            "Custom...": {"key": "weight", "val": "1.0"},
        },
        "presentation_example": {
            "title": "Add CommunityLeader Entity (Social Graph)",
            "scenario": "Demonstrate adding a new influencer or community leader and linking them to existing social groups.",
            "label": "Person",
            "id": "p_elena",
            "name": "Elena Rostova",
            "key1": "city",
            "val1": "Mumbai",
            "key2": "followers",
            "val2": "12500",
            "followup": "Connect p_elena -[:MEMBER_OF]-> g_ai"
        },
        "cypher_examples": {
            "-- Select an Educational Cypher Example --": "",
            "1. Match all people and their cities": "MATCH (p:Person) RETURN p.name, p.city, p.age",
            "2. Find friendships in network": "MATCH (p1:Person)-[:FRIENDS_WITH]->(p2:Person) RETURN p1.name, p2.name",
            "3. Group memberships by person": "MATCH (p:Person)-[m:MEMBER_OF]->(g:Group) RETURN p.name, g.name, m.role",
            "4. Shared topics of interest": "MATCH (p:Person)-[:INTERESTED_IN]->(t:Topic) RETURN p.name, t.name, t.domain",
            "5. Friend-of-friend multi-hop traversal": "MATCH (p1:Person)-[:FRIENDS_WITH]->(p2:Person)-[:FRIENDS_WITH]->(p3:Person) RETURN p1.name, p3.name",
            "6. Connect friends using CREATE": "MATCH (p1:Person {name: 'Alex'}), (p2:Person {name: 'Chris'}) CREATE (p1)-[:FRIENDS_WITH {since: 2024}]->(p2) RETURN p1, p2",
            "7. Count members in each group": "MATCH (p:Person)-[:MEMBER_OF]->(g:Group) RETURN g.name, count(p) AS total_members",
            "8. Delete relationship using DELETE": "MATCH (p:Person {name: 'Dan'})-[r:MEMBER_OF]->() DELETE r"
        }
    },
    "Financial Fraud Detection Ring": {
        "node_labels": ["Account", "Device", "IPAddress", "Merchant", "Custom..."],
        "default_properties": {
            "Account": {"id_prefix": "acc_", "name": "ACC-105", "key1": "owner", "val1": "David", "key2": "balance", "val2": "42000"},
            "Device": {"id_prefix": "dev_", "name": "DEV-MAC-9931", "key1": "device_id", "val1": "DEV-MAC-9931", "key2": "os", "val2": "iOS 17"},
            "IPAddress": {"id_prefix": "ip_", "name": "192.168.1.188", "key1": "ip", "val1": "192.168.1.188", "key2": "city", "val2": "Delhi"},
            "Merchant": {"id_prefix": "merch_", "name": "CryptoPay Gateway", "key1": "category", "val1": "Crypto", "key2": "risk_score", "val2": "95"},
            "Custom...": {"id_prefix": "custom_", "name": "Custom Entity", "key1": "risk_level", "val1": "HIGH", "key2": "status", "val2": "Flagged"},
        },
        "rel_types": ["USED_DEVICE", "LOGGED_FROM", "TRANSFER", "FLAGGED_WITH", "PAID_TO", "Custom..."],
        "default_rel_properties": {
            "USED_DEVICE": {"key": "last_login", "val": "2024-09-10"},
            "LOGGED_FROM": {"key": "timestamp", "val": "14:32:10"},
            "TRANSFER": {"key": "amount", "val": "8500"},
            "FLAGGED_WITH": {"key": "risk_score", "val": "88"},
            "PAID_TO": {"key": "amount", "val": "12000"},
            "Custom...": {"key": "weight", "val": "1.0"},
        },
        "presentation_example": {
            "title": "Add High-Risk Merchant Entity (Fraud Graph)",
            "scenario": "Demonstrate adding a high-risk crypto merchant or mule account to investigate laundering.",
            "label": "Merchant",
            "id": "merch_crypto",
            "name": "CryptoPay Ltd",
            "key1": "risk_level",
            "val1": "CRITICAL",
            "key2": "license",
            "val2": "Offshore-KYC-Bypass",
            "followup": "Connect acc_104 -[:TRANSFER {amount: 9500}]-> merch_crypto"
        },
        "cypher_examples": {
            "-- Select an Educational Cypher Example --": "",
            "1. Match all accounts and balances": "MATCH (a:Account) RETURN a.acc_no, a.owner, a.balance",
            "2. Detect circular money transfers (Fraud Ring)": "MATCH (a1:Account)-[t1:TRANSFER]->(a2:Account)-[t2:TRANSFER]->(a3:Account)-[t3:TRANSFER]->(a1) RETURN a1.owner, a2.owner, a3.owner, t1.amount, t2.amount, t3.amount",
            "3. Find shared devices across accounts": "MATCH (a:Account)-[:USED_DEVICE]->(d:Device) RETURN d.device_id, count(a) AS linked_accounts",
            "4. Trace accounts sharing same IP address": "MATCH (a:Account)-[:LOGGED_FROM]->(ip:IPAddress) RETURN ip.ip, ip.city, a.owner",
            "5. High-value transactions filter (WHERE amount >= 4500)": "MATCH (a1:Account)-[t:TRANSFER]->(a2:Account) WHERE t.amount >= 4500 RETURN a1.owner, a2.owner, t.amount",
            "6. Flag suspicious account with updated balance": "MATCH (a:Account {acc_no: 'ACC-104'}) SET a.balance = 0 RETURN a"
        }
    },
    "Blank / Empty Graph": {
        "node_labels": ["Entity", "User", "Concept", "Item", "Category", "Custom..."],
        "default_properties": {
            "Entity": {"id_prefix": "node_", "name": "Root Node", "key1": "type", "val1": "Base", "key2": "value", "val2": "100"},
            "User": {"id_prefix": "user_", "name": "Alex", "key1": "role", "val1": "Admin", "key2": "active", "val2": "True"},
            "Concept": {"id_prefix": "concept_", "name": "Graph Theory", "key1": "domain", "val1": "Mathematics", "key2": "difficulty", "val2": "Introductory"},
            "Item": {"id_prefix": "item_", "name": "Item A", "key1": "sku", "val1": "SKU001", "key2": "price", "val2": "49.99"},
            "Category": {"id_prefix": "cat_", "name": "Electronics", "key1": "code", "val1": "ELEC", "key2": "priority", "val2": "High"},
            "Custom...": {"id_prefix": "custom_", "name": "Custom Entity", "key1": "property_key", "val1": "property_val", "key2": "status", "val2": "Active"},
        },
        "rel_types": ["CONNECTED_TO", "RELATES_TO", "PART_OF", "DEPENDS_ON", "LINKED_WITH", "Custom..."],
        "default_rel_properties": {
            "CONNECTED_TO": {"key": "weight", "val": "1.0"},
            "RELATES_TO": {"key": "context", "val": "General"},
            "PART_OF": {"key": "order", "val": "1"},
            "DEPENDS_ON": {"key": "required", "val": "True"},
            "LINKED_WITH": {"key": "since", "val": "2024"},
            "Custom...": {"key": "weight", "val": "1.0"},
        },
        "presentation_example": {
            "title": "Add Custom Node from Scratch (Blank Graph)",
            "scenario": "Demonstrate creating custom domain entities and connecting them with directed relationships.",
            "label": "City",
            "id": "city_kgp",
            "name": "Kharagpur",
            "key1": "state",
            "val1": "West Bengal",
            "key2": "pin_code",
            "val2": "721302",
            "followup": "Add second node 'city_kolkata' and connect with CONNECTED_TO"
        },
        "cypher_examples": {
            "-- Select an Educational Cypher Example --": "",
            "1. Match all nodes in the graph": "MATCH (n) RETURN n",
            "2. Create custom node with properties": "CREATE (n:CustomEntity {id: 'c1', name: 'Sample Entity', priority: 'High'}) RETURN n",
            "3. Connect nodes with a relationship": "MATCH (a), (b) WHERE a.id = 'c1' AND b.id = 'c2' CREATE (a)-[:CONNECTED_TO {weight: 1.5}]->(b) RETURN a, b",
            "4. Detach delete all nodes (Reset)": "MATCH (n) DETACH DELETE n"
        }
    }
}

LABEL_COLORS = {
    "Student": "#2563EB",       # Blue
    "Course": "#F59E0B",        # Amber
    "Faculty": "#10B981",       # Emerald Green
    "Department": "#8B5CF6",    # Purple
    "Project": "#3B82F6",       # Bright Blue
    "Person": "#06B6D4",        # Cyan
    "Group": "#EC4899",         # Pink
    "Topic": "#14B8A6",         # Teal
    "Event": "#F43F5E",         # Rose
    "Account": "#EF4444",       # Red
    "Device": "#64748B",        # Slate
    "IPAddress": "#F97316",     # Orange
    "Merchant": "#DC2626",      # Crimson Red
    "ResearchLab": "#8B5CF6",   # Violet
    "City": "#0EA5E9",          # Sky Blue
    "Entity": "#6366F1",        # Indigo
    "User": "#2563EB",          # Blue
    "Concept": "#10B981",       # Emerald
    "Item": "#F59E0B",          # Amber
    "Category": "#A855F7"       # Purple
}
DEFAULT_NODE_COLOR = "#6366F1"  # Indigo

def get_node_color(label: str) -> str:
    """Returns color for a node label with deterministic fallback."""
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    palette = ["#6366F1", "#EC4899", "#14B8A6", "#F59E0B", "#10B981", "#8B5CF6", "#06B6D4", "#F97316", "#3B82F6"]
    return palette[abs(hash(label)) % len(palette)]


def get_primary_label(labels: Any) -> str:
    """Returns the primary label for a node, prioritizing domain entities (e.g. Student, Faculty) over generic labels."""
    if not labels:
        return "Entity"
    lbl_list = list(labels)
    # Prefer specific domain entities like Student, Faculty in academic/university context
    priority = ["Student", "Faculty", "Course", "Department", "Project", "Account", "Device", "Merchant", "Group", "Topic"]
    for pref in priority:
        if pref in lbl_list:
            return pref
    return sorted(lbl_list)[0]


def render_graph_figure(graph: PropertyGraph,
                         matched_node_ids: Optional[List[str]] = None,
                         matched_rel_ids: Optional[List[str]] = None,
                         layout_algorithm: str = "Spring (Force-Directed)",
                         node_label_mode: str = "Name / Label",
                         theme: str = "auto") -> go.Figure:
    """
    Renders an interactive 2D graph visualization using NetworkX for layout
    and Plotly for interactive rendering with hovercards and arrows.
    """
    fig = go.Figure()

    if not graph.nodes:
        fig.add_annotation(
            text="Graph is currently empty. Add nodes or load a preset dataset!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#64748B")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    # Build NetworkX representation for coordinates
    G = graph.to_networkx()

    # Layout computation
    n_count = len(G.nodes())
    if layout_algorithm == "Circular":
        pos = nx.circular_layout(G)
    elif layout_algorithm == "Kamada-Kawai":
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=max(0.6, 2.5 / np.sqrt(max(1, n_count))))
    elif layout_algorithm == "Shell":
        label_groups = {}
        for nid, node in graph.nodes.items():
            primary_lbl = get_primary_label(node.labels)
            label_groups.setdefault(primary_lbl, []).append(nid)
        pos = nx.shell_layout(G, nlist=list(label_groups.values()))
    else:  # Default Spring
        pos = nx.spring_layout(G, seed=42, k=max(0.7, 3.0 / np.sqrt(max(1, n_count))), iterations=60)

    matched_nids_set = set(matched_node_ids) if matched_node_ids else set()
    matched_rids_set = set(matched_rel_ids) if matched_rel_ids else set()

    # 1. Edge Line Traces
    edge_x = []
    edge_y = []
    mid_x = []
    mid_y = []
    mid_text = []
    mid_color = []

    for rid, rel in graph.relationships.items():
        if rel.source not in pos or rel.target not in pos:
            continue
        x0, y0 = pos[rel.source]
        x1, y1 = pos[rel.target]

        # Line segment
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Directional point
        mx = x0 + 0.60 * (x1 - x0)
        my = y0 + 0.60 * (y1 - y0)
        mid_x.append(mx)
        mid_y.append(my)

        is_matched = rid in matched_rids_set
        color = "#F59E0B" if is_matched else "#64748B"
        mid_color.append(color)
        mid_text.append(f"<b>{rel.type}</b>")

    # Base Edge Lines
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=2, color="#94A3B8"),
        hoverinfo="none",
        showlegend=False
    ))

    # Relationship Labels & Direction Indicators
    if mid_x:
        fig.add_trace(go.Scatter(
            x=mid_x, y=mid_y,
            mode="text+markers",
            marker=dict(size=10, symbol="triangle-up", color=mid_color, line=dict(width=1, color="#64748B")),
            text=mid_text,
            textposition="top center",
            textfont=dict(size=10, color="#475569", family="Arial, sans-serif"),
            hoverinfo="none",
            showlegend=False
        ))

    # 2. Node Traces (Grouped by Primary Label for clean legend)
    nodes_by_label: Dict[str, List[str]] = {}
    for nid, node in graph.nodes.items():
        primary_label = get_primary_label(node.labels)
        nodes_by_label.setdefault(primary_label, []).append(nid)

    for label_name, nids in nodes_by_label.items():
        nx_coords = []
        ny_coords = []
        display_texts = []
        sizes = []
        line_widths = []
        line_colors = []

        base_color = get_node_color(label_name)

        for nid in nids:
            if nid not in pos:
                continue
            x, y = pos[nid]
            nx_coords.append(x)
            ny_coords.append(y)

            node = graph.nodes[nid]
            is_matched = nid in matched_nids_set

            if is_matched:
                sizes.append(45)
                line_widths.append(4)
                line_colors.append("#FDE047")  # Bright Yellow Highlight
            else:
                sizes.append(36)
                line_widths.append(2.5)
                line_colors.append("#FFFFFF")

            # Display text
            if node_label_mode == "Name / Label":
                display_texts.append(f"<b>{node.display_name()}</b><br><i>:{label_name}</i>")
            elif node_label_mode == "Name Only":
                display_texts.append(f"<b>{node.display_name()}</b>")
            elif node_label_mode == "Node ID":
                display_texts.append(f"<b>{nid}</b>")
            elif node_label_mode == "Label Only":
                display_texts.append(f"<b>:{label_name}</b>")
            else:
                display_texts.append("")

        fig.add_trace(go.Scatter(
            x=nx_coords, y=ny_coords,
            mode="markers+text",
            name=f":{label_name} ({len(nids)})",
            marker=dict(
                size=sizes,
                color=base_color,
                line=dict(width=line_widths, color=line_colors),
                opacity=0.95
            ),
            text=display_texts,
            textposition="bottom center",
            textfont=dict(size=12, color="#0F172A", family="Arial, sans-serif"),
            hoverinfo="none"
        ))

    # Matched highlight legend indicator if any
    if matched_node_ids:
        fig.add_annotation(
            text=f"Matched Subgraph: {len(matched_node_ids)} Node(s) highlighted",
            xref="paper", yref="paper",
            x=0.01, y=0.99, showarrow=False,
            bgcolor="#FEF08A",
            font=dict(size=12, color="#854D0E", family="Arial, sans-serif"),
            bordercolor="#FACC15",
            borderwidth=2,
            borderpad=6,
            opacity=0.95
        )

    fig.update_layout(
        title=dict(text="<b>Graph Topology</b>", font=dict(size=18, family="Arial, sans-serif", color="#0F172A")),
        hovermode=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=720,
        margin=dict(l=15, r=15, t=50, b=20),
        plot_bgcolor="rgba(248,250,252,0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#0F172A"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1
        )
    )

    return fig


@st.cache_data
def load_vis_network_js() -> str:
    """Loads the local standalone vis-network library to guarantee 100% offline graph rendering."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "vis-network.min.js")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    return ""


def render_interactive_graph_canvas(graph: PropertyGraph, 
                                    matched_node_ids: Optional[List[str]] = None,
                                    matched_rel_ids: Optional[List[str]] = None) -> None:
    """Renders a high-performance interactive Property Graph canvas with live physics & simulation animations."""
    
    nodes_data = []
    edges_data = []
    
    matched_nids = set(matched_node_ids) if matched_node_ids else set()
    matched_rids = set(matched_rel_ids) if matched_rel_ids else set()
    
    for nid, node in graph.nodes.items():
        primary_label = get_primary_label(node.labels)
        base_color = get_node_color(primary_label)
        
        is_matched = nid in matched_nids
        border_width = 4 if is_matched else 2
        border_color = "#F59E0B" if is_matched else "#FFFFFF"
        
        # Build hover title (HTML)
        labels_str = ":" + ":".join(sorted(node.labels))
        prop_lines = "".join([f"<tr><td style='padding-right:8px; color:#475569;'><b>{k}</b></td><td style='color:#0f172a;'>{v}</td></tr>" for k, v in node.properties.items()])
        title_html = f"<div style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; padding:6px; min-width:140px;'><b style='color:#0f172a; font-size:14px;'>{node.display_name()}</b><br><span style='color:#0284c7; font-weight:600; font-size:12px;'>{labels_str}</span><br><span style='color:#64748b; font-size:11px;'>ID: {nid}</span>"
        if prop_lines:
            title_html += f"<hr style='margin:5px 0; border:0; border-top:1px solid #e2e8f0;'><table style='font-size:11px; width:100%;'>{prop_lines}</table>"
        title_html += "</div>"
        
        nodes_data.append({
            "id": nid,
            "label": f"<b>{node.display_name()}</b>\n<i>:{primary_label}</i>",
            "title": title_html,
            "color": {
                "background": base_color,
                "border": border_color,
                "highlight": {"background": base_color, "border": "#F59E0B"},
                "hover": {"background": base_color, "border": "#0284C7"}
            },
            "borderWidth": border_width,
            "borderWidthSelected": 4,
            "shape": "dot",
            "size": 28 if not is_matched else 34,
            "font": {"size": 12, "color": "#0F172A", "face": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", "multi": "html", "align": "center"}
        })
        
    for rid, rel in graph.relationships.items():
        is_matched = rid in matched_rids
        edge_color = "#F59E0B" if is_matched else "#94A3B8"
        
        prop_lines = "".join([f"<tr><td style='padding-right:8px; color:#475569;'><b>{k}</b></td><td style='color:#0f172a;'>{v}</td></tr>" for k, v in rel.properties.items()])
        title_html = f"<div style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; padding:6px;'><b style='color:#0f172a; font-size:13px;'>[:{rel.type}]</b><br><span style='color:#64748b; font-size:11px;'>{rel.source} &rarr; {rel.target}</span>"
        if prop_lines:
            title_html += f"<hr style='margin:5px 0; border:0; border-top:1px solid #e2e8f0;'><table style='font-size:11px; width:100%;'>{prop_lines}</table>"
        title_html += "</div>"
        
        edges_data.append({
            "id": rid,
            "from": rel.source,
            "to": rel.target,
            "label": f"  {rel.type}  ",
            "title": title_html,
            "color": {"color": edge_color, "highlight": "#F59E0B", "hover": "#0284C7"},
            "width": 2 if not is_matched else 3.5,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
            "font": {"size": 10, "color": "#475569", "face": "-apple-system, BlinkMacSystemFont, sans-serif", "background": "rgba(255,255,255,0.92)", "strokeWidth": 0, "align": "middle"},
            "smooth": {"type": "continuous", "roundness": 0.15}
        })

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)
    
    # Load vis-network JS locally to guarantee offline rendering without unpkg dependency
    local_vis_js = load_vis_network_js()
    if local_vis_js:
        script_block = "<script type=\"text/javascript\">\n" + local_vis_js + "\n</script>"
    else:
        script_block = "<script type=\"text/javascript\" src=\"https://unpkg.com/vis-network/standalone/umd/vis-network.min.js\"></script>"

    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    __VIS_NETWORK_SCRIPT_TAG__
    <style type="text/css">
        body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; overflow: hidden; background: #f8fafc; }
        #canvas-wrapper { position: relative; width: 100%; height: 720px; }
        #mynetwork {
            width: 100%;
            height: 720px;
            border: 1px solid #cbd5e1;
            background: radial-gradient(circle at center, #ffffff 0%, #f1f5f9 100%);
            border-radius: 8px;
            box-sizing: border-box;
        }
        .vis-tooltip {
            background-color: white !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 6px !important;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.12) !important;
            color: #1e293b !important;
            padding: 0 !important;
            pointer-events: none;
            z-index: 1000 !important;
        }
        #controls {
            position: absolute;
            bottom: 12px;
            left: 12px;
            z-index: 50;
            display: flex;
            gap: 6px;
            background: rgba(255, 255, 255, 0.95);
            padding: 5px 8px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.1);
            border: 1px solid #cbd5e1;
            align-items: center;
        }
        .ctrl-btn {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 5px;
            padding: 5px 9px;
            cursor: pointer;
            font-size: 12px;
            color: #334155;
            font-weight: 600;
            transition: all 0.15s ease-in-out;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .ctrl-btn:hover { background: #0284c7; border-color: #0284c7; color: #ffffff; }
        .ctrl-btn.active { background: #0284c7; color: #ffffff; border-color: #0284c7; }
        
        #legend {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 50;
            background: rgba(255, 255, 255, 0.94);
            padding: 8px 12px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
            border: 1px solid #cbd5e1;
            max-width: 170px;
            font-size: 11px;
            max-height: 190px;
            overflow-y: auto;
        }
        .legend-title { font-weight: 700; margin-bottom: 5px; color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 3px; text-transform: uppercase; font-size: 10px; letter-spacing: 0.05em; }
        .legend-item { display: flex; align-items: center; margin-bottom: 4px; justify-content: space-between; }
        .legend-label-group { display: flex; align-items: center; gap: 5px; }
        .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; border: 1px solid rgba(0,0,0,0.1); }
        .badge { background: #e0f2fe; color: #0369a1; padding: 1px 5px; border-radius: 999px; font-size: 9px; font-weight: 700; }
        .rel-badge { background: #f1f5f9; border: 1px solid #cbd5e1; color: #475569; padding: 1px 5px; border-radius: 4px; font-size: 9px; font-weight: 600; font-family: monospace; }
    </style>
</head>
<body>
    <div id="canvas-wrapper">
        <div id="mynetwork"></div>
        
        <div id="legend">
            <div class="legend-title">Labels</div>
            <div id="node-legend-container"></div>
            <div class="legend-title" style="margin-top: 8px;">Relationships</div>
            <div id="rel-legend-container"></div>
        </div>
        
        <div id="controls">
            <button class="ctrl-btn" id="physics-btn" onclick="togglePhysics()" title="Toggle Physics Bounce">Freeze</button>
            <button class="ctrl-btn" id="pulse-btn" onclick="togglePulseAnimation()" title="Animate Path Traversal Across Nodes">Pulse Traversal</button>
            <button class="ctrl-btn" onclick="scatterAndSnap()" title="Scatter Outward & Elastic Snap Back">Scatter</button>
            <button class="ctrl-btn" onclick="centerGraph()" title="Fit to View">Center</button>
            <button class="ctrl-btn" onclick="zoom(0.2)" title="Zoom In">+</button>
            <button class="ctrl-btn" onclick="zoom(-0.2)" title="Zoom Out">-</button>
        </div>
    </div>

    <script type="text/javascript">
        window.onerror = function(msg, url, line) {
            var c = document.getElementById('mynetwork');
            if (c) {
                c.innerHTML = '<div style="padding:20px; color:#ef4444; font-family:sans-serif;"><b>Visualizer error:</b> ' + msg + '</div>';
            }
        };

        var rawNodes = __NODES_JSON_PLACEHOLDER__;
        var rawEdges = __EDGES_JSON_PLACEHOLDER__;

        var nodes = new vis.DataSet(rawNodes);
        var edges = new vis.DataSet(rawEdges);

        var container = document.getElementById('mynetwork');
        var data = { nodes: nodes, edges: edges };
        var options = {
            physics: {
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -75,
                    centralGravity: 0.018,
                    springLength: 105,
                    springConstant: 0.08,
                    damping: 0.45,
                    avoidOverlap: 0.45
                },
                stabilization: { iterations: 120, updateInterval: 25 }
            },
            interaction: {
                hover: true,
                tooltipDelay: 60,
                zoomView: true,
                dragView: true,
                dragNodes: true,
                navigationButtons: false
            },
            layout: {
                improvedLayout: true
            }
        };
        var network = new vis.Network(container, data, options);
        
        // Build Legend dynamically
        var nodeLegendContainer = document.getElementById('node-legend-container');
        var relLegendContainer = document.getElementById('rel-legend-container');
        
        var labelCounts = {};
        var labelColors = {};
        nodes.forEach(function(n) {
            var parts = n.label.split("<i>:");
            var lbl = (parts.length > 1) ? parts[1].split("</i>")[0] : "Entity";
            labelCounts[lbl] = (labelCounts[lbl] || 0) + 1;
            labelColors[lbl] = n.color.background;
        });
        
        var relCounts = {};
        edges.forEach(function(e) {
            var type = e.label.trim();
            relCounts[type] = (relCounts[type] || 0) + 1;
        });
        
        Object.keys(labelCounts).sort().forEach(function(lbl) {
            var div = document.createElement('div');
            div.className = 'legend-item';
            div.innerHTML = '<div class="legend-label-group"><span class="dot" style="background-color: ' + labelColors[lbl] + ';"></span> <span>' + lbl + '</span></div> <span class="badge">' + labelCounts[lbl] + '</span>';
            nodeLegendContainer.appendChild(div);
        });
        
        Object.keys(relCounts).sort().forEach(function(type) {
            var div = document.createElement('div');
            div.className = 'legend-item';
            div.innerHTML = '<span class="rel-badge">' + type + '</span> <span class="badge">' + relCounts[type] + '</span>';
            relLegendContainer.appendChild(div);
        });
        
        function zoom(scale) {
            var newScale = network.getScale() * (1 + scale);
            network.moveTo({ scale: newScale, animation: { duration: 250, easingFunction: 'easeInOutQuad' } });
        }
        
        function centerGraph() {
            network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
        }
        
        var physicsEnabled = true;
        function togglePhysics() {
            physicsEnabled = !physicsEnabled;
            network.setOptions({ physics: { enabled: physicsEnabled } });
            var btn = document.getElementById('physics-btn');
            btn.innerHTML = physicsEnabled ? 'Freeze' : 'Live Physics';
            btn.classList.toggle('active', !physicsEnabled);
        }

        // Pulse / Path Traversal Simulation Animation
        var pulseTimer = null;
        function togglePulseAnimation() {
            var btn = document.getElementById('pulse-btn');
            if (pulseTimer) {
                clearInterval(pulseTimer);
                pulseTimer = null;
                btn.innerHTML = 'Pulse Traversal';
                btn.classList.remove('active');
                // Restore base sizes
                var resets = [];
                nodes.forEach(function(n) {
                    resets.push({ id: n.id, size: 28, borderWidth: 2 });
                });
                nodes.update(resets);
                return;
            }
            btn.innerHTML = 'Stop Pulse';
            btn.classList.add('active');
            var allIds = nodes.getIds();
            if (!allIds || allIds.length === 0) return;
            var step = 0;
            pulseTimer = setInterval(function() {
                var targetId = allIds[step % allIds.length];
                var updates = [];
                nodes.forEach(function(n) {
                    if (n.id === targetId) {
                        updates.push({ id: n.id, size: 40, borderWidth: 5 });
                    } else {
                        updates.push({ id: n.id, size: 28, borderWidth: 2 });
                    }
                });
                nodes.update(updates);
                step++;
            }, 380);
        }

        // Scatter & Snap Force Elasticity Simulation
        function scatterAndSnap() {
            network.setOptions({ physics: { enabled: true } });
            var allIds = nodes.getIds();
            allIds.forEach(function(id) {
                var angle = Math.random() * Math.PI * 2;
                var dist = 200 + Math.random() * 200;
                network.moveNode(id, Math.cos(angle) * dist, Math.sin(angle) * dist);
            });
            setTimeout(function() {
                network.fit({ animation: { duration: 1000, easingFunction: 'easeInOutQuad' } });
            }, 300);
        }
    </script>
</body>
</html>
"""

    html_code = html_template.replace("__VIS_NETWORK_SCRIPT_TAG__", script_block)
    html_code = html_code.replace("__NODES_JSON_PLACEHOLDER__", nodes_json)
    html_code = html_code.replace("__EDGES_JSON_PLACEHOLDER__", edges_json)
    
    components.html(html_code, height=750)


# ======================================================================================
# 5. SECTION RENDERERS

# ======================================================================================

def go_to_simulation():
    """Route user to the simulation section."""
    st.session_state["requested_section"] = "Simulation"


def go_to_theory():
    """Route user to the theory section."""
    st.session_state["requested_section"] = "Theory"


def render_purpose_section():
    """Renders the Purpose section: Why this experiment, What problem it solves, History, and Where all it is used."""
    st.markdown(f"""
        <div class="purpose-header">
            <div class="hero-eyebrow">EXPERIMENT 08</div>
            <h1>Create & Manage a Graph Database</h1>
            <p class="subtitle">A paradigm shift from rigid tabular SQL tables to connected native Property Graphs with Index-Free Adjacency.</p>
        </div>
    """, unsafe_allow_html=True)

    tab_why, tab_problem, tab_history, tab_usecases = st.tabs([
        "1. Why This Experiment?",
        "2. The Problem Solved",
        "3. Evolution & History",
        "4. Where All It's Used"
    ])

    with tab_why:
        col_w1, col_w2 = st.columns([1.1, 0.9])
        with col_w1:
            st.markdown("""
                <div class="purpose-section-card" style="height: 100%;">
                    <h3>Connecting the Networked World</h3>
                    <p>
                        In modern software systems, <strong>the highest-value intelligence lives in the connections between entities</strong>. 
                        Whether it is social friendships, transactional money flows across banks, or medical interactions between genes and drugs, 
                        real-world data is inherently a web, not a flat spreadsheet.
                    </p>
                    <p>
                        For over 40 years, computer science curricula emphasized <strong>Relational Database Management Systems (RDBMS)</strong>. 
                        While relational tables excel at flat accounting ledgers, they degrade exponentially when relationships multiply.
                    </p>
                    <div class="highlight-box">
                        <strong>The Paradigm Shift:</strong> Move from <em>Tabular Thinking</em> (tables, rows, foreign key joins) to 
                        <em>Graph Thinking</em> (nodes, directed edges, rich properties, and pointer-chasing traversals).
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_w2:
            st.markdown("""
                <div class="purpose-section-card" style="height: 100%;">
                    <h3>Core Learning Outcomes</h3>
                    <ul>
                        <li><strong>Hands-on Visual Modeling:</strong> Design entities as Nodes and typed connections as Relationships with properties.</li>
                        <li><strong>Declarative Query Mastery:</strong> Learn Cypher—the standardized ASCII-art query language: <code>(n)-[:REL]->(m)</code>.</li>
                        <li><strong>Architectural Insight:</strong> Understand how native graph storage uses direct memory pointers (Index-Free Adjacency) for O(1) step traversal.</li>
                        <li><strong>Industrial Competence:</strong> Gain skills directly applicable to AI Knowledge Graphs (GraphRAG), fraud detection, and recommendation systems.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    with tab_problem:
        st.markdown("""
            <div class="purpose-section-card">
                <h3>Overcoming the Relational Bottleneck</h3>
                <div class="vs-grid" style="margin-top: 0.8rem;">
                    <div class="vs-box bad">
                        <h4 style="color:#ef4444; margin-top:0; font-size:1.25rem;">Relational Databases: The JOIN Explosion</h4>
                        <p>SQL splits data across isolated tables. Connecting entities requires index-lookup <code>JOIN</code> operations on foreign keys.</p>
                        <ul>
                            <li><strong>Exponential Slowdown:</strong> Finding "friends of friends of friends" needs 3–5 joins, exploding query time exponentially (<em>O(N<sup>k</sup>)</em>).</li>
                            <li><strong>Memory Spikes & Freezes:</strong> Millions of rows force full table/index scans, exhausting RAM and causing query timeouts.</li>
                            <li><strong>Brittle Schema Migrations:</strong> Adding relationship types requires risky <code>ALTER TABLE</code> schema changes with production downtime.</li>
                        </ul>
                    </div>
                    <div class="vs-box good">
                        <h4 style="color:#10b981; margin-top:0; font-size:1.25rem;">Graph Databases: Index-Free Adjacency (IFA)</h4>
                        <p>Graph DBMS engines treat relationships as first-class physical pointers in storage and memory.</p>
                        <ul>
                            <li><strong>O(1) Constant-Time Traversal:</strong> Every node holds direct physical memory pointers to adjacent nodes. Hopping takes constant time!</li>
                            <li><strong>Predictable Real-Time Latency:</strong> Traversal speed depends only on the subgraph traversed, independent of total database size (10K or 100M nodes).</li>
                            <li><strong>Flexible & Schema-Optional:</strong> Add new node labels, relationships, and custom properties on the fly without breaking schemas.</li>
                        </ul>
                    </div>
                </div>
                <div class="highlight-box" style="margin-top: 1rem;">
                    <strong>Key Architectural Law:</strong> Relational joins search global indexes on every hop. Graph databases eliminate index lookups entirely during traversal by chasing direct memory pointers.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with tab_history:
        st.markdown("""
            <div class="purpose-section-card">
                <h3>Three Centuries of Graph Innovation</h3>
                <div class="usecase-grid" style="grid-template-columns: repeat(2, 1fr); margin-top: 0.8rem;">
                    <div class="usecase-card">
                        <div class="timeline-year" style="color:var(--accent); font-weight:800; font-size:1.25rem;">1736</div>
                        <strong>Leonhard Euler & Königsberg Bridges</strong>
                        <p>Leonhard Euler proved that traversing Königsberg's 7 bridges without retracing steps was impossible. By abstracting landmasses to vertices and bridges to edges, <strong>Graph Theory</strong> was born.</p>
                    </div>
                    <div class="usecase-card">
                        <div class="timeline-year" style="color:var(--accent); font-weight:800; font-size:1.25rem;">1970 – 1990s</div>
                        <strong>Relational Hegemony & SQL Standards</strong>
                        <p>E.F. Codd published the Relational Model at IBM. Tables, foreign keys, and SQL dominated business computing for decades, establishing standard transactional guarantees.</p>
                    </div>
                    <div class="usecase-card">
                        <div class="timeline-year" style="color:var(--accent); font-weight:800; font-size:1.25rem;">2000s</div>
                        <strong>The Web, PageRank & NoSQL Wave</strong>
                        <p>Google's PageRank algorithm, LinkedIn's Economic Graph, and Facebook proved the value of networks. Relational join limits sparked the NoSQL wave (Key-Value, Document, Columnar, Graph).</p>
                    </div>
                    <div class="usecase-card">
                        <div class="timeline-year" style="color:var(--accent); font-weight:800; font-size:1.25rem;">2010s – Today</div>
                        <strong>Property Graphs & ISO GQL Standard</strong>
                        <p>The Labeled Property Graph (LPG) model matured with declarative Cypher. In 2024, the International Organization for Standardization ratified <strong>ISO/IEC 39075:2024 GQL</strong>, the first new ISO database query language since SQL (1987).</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with tab_usecases:
        st.markdown("""
            <div class="purpose-section-card">
                <h3>Global Production Deployments</h3>
                <div class="usecase-grid" style="grid-template-columns: repeat(3, 1fr); margin-top: 0.8rem;">
                    <div class="usecase-card">
                        <strong>Fraud & Money Laundering</strong>
                        <p>Detects synthetic identities, shared phone numbers across cards, and circular fund routing across banks in real time before wire release.</p>
                    </div>
                    <div class="usecase-card">
                        <strong>Real-Time Recommendation</strong>
                        <p>Powers recommendation feeds on Amazon, Netflix, and Spotify through real-time multi-hop collaborative path exploration.</p>
                    </div>
                    <div class="usecase-card">
                        <strong>Knowledge Graphs & AI (GraphRAG)</strong>
                        <p>Connects LLMs with verified factual knowledge graphs to eliminate AI hallucinations and provide verifiable source citations.</p>
                    </div>
                    <div class="usecase-card">
                        <strong>Healthcare & Drug Discovery</strong>
                        <p>Maps interactions between diseases, genes, proteins, and chemical compounds to accelerate pharmaceutical drug repurposing.</p>
                    </div>
                    <div class="usecase-card">
                        <strong>Cybersecurity & IAM</strong>
                        <p>Audits Active Directory and cloud permissions (e.g., BloodHound) to detect hidden privilege escalation attack paths before breach.</p>
                    </div>
                    <div class="usecase-card">
                        <strong>Supply Chain Resilience</strong>
                        <p>Maps tier-1 to tier-N supplier dependencies to detect single points of failure and reroute critical logistics during disruptions.</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Next steps call to action
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Proceed to Theory & Architecture", type="primary", use_container_width=True, key="purpose_to_theory_btn"):
            st.session_state["requested_section"] = "Theory"
            st.rerun()
    with col2:
        if st.button("Launch Interactive Simulation", use_container_width=True, key="purpose_to_sim_btn"):
            st.session_state["requested_section"] = "Simulation"
            st.rerun()


def render_graph_concept_diagram():
    """Adds one compact visual explanation to the textbook introduction."""
    st.markdown("""
        <div class="concept-diagram">
            <div><span class="diagram-node">NODE</span><small>entity</small></div>
            <div class="diagram-arrow">→<small>relationship</small></div>
            <div><span class="diagram-node accent">NODE</span><small>entity</small></div>
            <div class="diagram-arrow">→<small>traversal</small></div>
            <div><span class="diagram-node warm">NODE</span><small>entity</small></div>
        </div>
    """, unsafe_allow_html=True)


def render_theory_section():
    """Renders Section: Theory, Background, Architecture, Cypher, and Procedure."""
    st.markdown(f"""
        <div class="purpose-header" style="padding:1.2rem 1.8rem; margin-bottom:1.2rem;">
            <div class="hero-eyebrow"><span class="hero-dot"></span>THEORETICAL FRAMEWORK · {EXPERIMENT_CONFIG['lab_code']}</div>
            <h1 style="font-size:1.8rem !important; margin:0.2rem 0 !important;">Graph Databases & Cypher Foundations</h1>
            <p class="subtitle"><b>Aim:</b> {THEORY_CONTENT['aim']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs for better organization
    tabs = st.tabs([
        "Introduction", 
        "RDBMS vs Graph", 
        "Graph Architecture", 
        "Cypher & CRUD", 
        "Procedure", 
        "Glossary"
    ])

    with tabs[0]:
        render_graph_concept_diagram()
        st.markdown(THEORY_CONTENT["introduction"])
    
    with tabs[1]:
        st.markdown(THEORY_CONTENT["rdbms_vs_graph"])
        st.markdown('<div class="manual-intro"><strong>Key takeaway</strong><br>Index-Free Adjacency (IFA) keeps graph traversal proportional to the path being explored, rather than the total database size.</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown(THEORY_CONTENT["graph_architecture"])

    with tabs[3]:
        st.markdown(THEORY_CONTENT["cypher_crud"])

    with tabs[4]:
        st.markdown(THEORY_CONTENT["setup_procedure"])
        st.divider()
        st.subheader("Step-by-Step Experimental Procedure")
        for step in THEORY_CONTENT["procedure"]:
            st.write(f"- {step}")
        st.divider()
        st.markdown('<div class="manual-label">Precautions and Common Mistakes</div>', unsafe_allow_html=True)
        st.markdown(THEORY_CONTENT["precautions"])

    with tabs[5]:
        st.markdown("### Comprehensive Key Terminology Reference")
        glossary_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term", "Formal Definition & Operational Role"]
        )
        st.dataframe(glossary_df, use_container_width=True, hide_index=True)


def render_simulation_section():
    """Renders Section: Interactive Graph Workstation, Visual Controls, Cypher Console, Topology Canvas & Activity."""
    graph: PropertyGraph = st.session_state["graph"]
    engine: CypherEngine = st.session_state["cypher_engine"]

    preset_options = [
        "University Academic Knowledge Graph (Default)",
        "Social Network & Friendships",
        "Financial Fraud Detection Ring",
        "Blank / Empty Graph"
    ]
    current_preset_idx = st.session_state.get("preset_index", 0)
    active_preset_name = st.session_state.get("active_preset", preset_options[current_preset_idx])
    schema = PRESET_SCHEMAS.get(active_preset_name, PRESET_SCHEMAS["University Academic Knowledge Graph (Default)"])
    metrics = graph.get_metrics()

    # ----------------------------------------------------------------------------------
    # TOP CONTROL BAR: Grouped Presets, Live Metrics & Reset
    # ----------------------------------------------------------------------------------
    with st.container(border=True):
        col_pres, col_load, col_metrics, col_reset = st.columns([2.4, 1.2, 2.8, 1.0])

        with col_pres:
            preset_choice = st.selectbox(
                "Domain Preset:",
                options=preset_options,
                index=current_preset_idx,
                key="simulation_preset_select",
                label_visibility="collapsed"
            )

        with col_load:
            if st.button("Load Preset", use_container_width=True, type="primary"):
                if preset_choice.startswith("University"):
                    graph.load_university_graph()
                    st.session_state["preset_index"] = 0
                elif preset_choice.startswith("Social"):
                    graph.load_social_graph()
                    st.session_state["preset_index"] = 1
                elif preset_choice.startswith("Financial"):
                    graph.load_fraud_graph()
                    st.session_state["preset_index"] = 2
                else:
                    graph.clear()
                    st.session_state["preset_index"] = 3

                st.session_state["active_preset"] = preset_choice
                st.session_state["matched_node_ids"] = []
                st.session_state["matched_rel_ids"] = []
                st.session_state["last_cypher_result"] = None
                log_activity("Preset", f"Loaded '{preset_choice}'", "info")
                for k in ["node_label_sel", "custom_label_inp", "node_id_inp", "node_name_inp", 
                          "prop_key1_inp", "prop_val1_inp", "prop_key2_inp", "prop_val2_inp",
                          "rel_type_sel", "custom_rel_inp", "rel_prop_k_inp", "rel_prop_v_inp"]:
                    st.session_state.pop(k, None)
                st.toast(f"Loaded '{preset_choice}'!")
                st.rerun()

        with col_metrics:
            st.markdown(
                f'<div class="metric-chips-row" style="margin-top:2px;">'
                f'<span class="metric-chip"><strong>{metrics["num_nodes"]}</strong> nodes</span>'
                f'<span class="metric-chip"><strong>{metrics["num_relationships"]}</strong> rels</span>'
                f'<span class="metric-chip"><strong>{metrics["num_labels"]}</strong> labels</span>'
                f'<span class="metric-chip"><strong>{metrics["density"]}</strong> density</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        with col_reset:
            if st.button("Clear All", type="secondary", use_container_width=True):
                graph.clear()
                st.session_state["matched_node_ids"] = []
                st.session_state["matched_rel_ids"] = []
                st.session_state["last_cypher_result"] = None
                log_activity("Reset", "Cleared entire graph", "warning")
                st.toast("Graph cleared.")
                st.rerun()

    # ----------------------------------------------------------------------------------
    # TWO-COLUMN SINGLE-PAPER WORKSTATION
    # Left Column: Tools & Console (45%) | Right Column: Topology Canvas & Animation (55%)
    # ----------------------------------------------------------------------------------
    col_tools, col_canvas = st.columns([5.0, 7.0])

    with col_tools:
        tab_cypher, tab_builder, tab_logger = st.tabs([
            "Cypher Console",
            "Visual Builder",
            "Observation Log"
        ])

        with tab_cypher:
            cypher_examples = schema["cypher_examples"]
            c_ex, c_load_q = st.columns([3.2, 1.2])
            with c_ex:
                selected_example = st.selectbox(
                    "Query Examples:",
                    options=list(cypher_examples.keys()),
                    index=0,
                    key=f"cypher_ex_{active_preset_name}",
                    label_visibility="collapsed"
                )
            with c_load_q:
                if st.button("Paste", use_container_width=True):
                    if cypher_examples.get(selected_example):
                        st.session_state["current_query_input"] = cypher_examples[selected_example]
                        st.rerun()

            default_cypher = list(cypher_examples.values())[1] if len(cypher_examples) > 1 else "MATCH (n) RETURN n"
            query_input = st.text_area(
                "Cypher Statement:",
                value=st.session_state.get("current_query_input", default_cypher),
                height=80,
                label_visibility="collapsed",
                help="Type Cypher query. Example: MATCH (n)-[r]->(m) RETURN n, r, m"
            )
            st.session_state["current_query_input"] = query_input

            if st.button("Run Cypher Query", type="primary", use_container_width=True):
                res = engine.execute(query_input)
                st.session_state["last_cypher_result"] = res
                st.session_state["matched_node_ids"] = res["matched_node_ids"]
                st.session_state["matched_rel_ids"] = res["matched_rel_ids"]

                trial_record = {
                    "Trial #": len(st.session_state["trials"]) + 1,
                    "Operation": "Cypher Query",
                    "Query / Action": query_input[:65],
                    "Result": res["message"][:40],
                    "Status": "Success" if res["success"] else "Failed",
                    "Timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state["trials"].append(trial_record)
                log_activity("Cypher", query_input.strip().replace('\n', ' ')[:30], "query" if res["success"] else "error")

            last_res = st.session_state.get("last_cypher_result")
            if last_res:
                if last_res["success"]:
                    st.success(f"**Query Succeeded** ({last_res['execution_time_ms']} ms): {last_res['message']}")
                    df = last_res["dataframe"]
                    if not df.empty:
                        st.dataframe(df, use_container_width=True, hide_index=True, height=140)
                else:
                    st.error(f"**Query Failed** ({last_res['execution_time_ms']} ms): {last_res['message']}")

        with tab_builder:
            sub_tab_node, sub_tab_rel, sub_tab_set, sub_tab_del = st.tabs([
                "Add Node", "Add Rel", "SET Prop", "Delete"
            ])

            # SUB-TAB A: ADD NODE
            with sub_tab_node:
                col_lbl, col_id = st.columns(2)
                with col_lbl:
                    label_options = schema["node_labels"]
                    def_lbl_idx = 0
                    if "node_label_sel" in st.session_state and st.session_state["node_label_sel"] in label_options:
                        def_lbl_idx = label_options.index(st.session_state["node_label_sel"])
                    node_label = st.selectbox("Label:", label_options, index=def_lbl_idx, key="node_label_sel")
                    if node_label == "Custom...":
                        custom_lbl_val = st.session_state.get("custom_label_inp", "Topic")
                        custom_node_label = st.text_input("Custom Label:", value=custom_lbl_val, key="custom_label_inp")
                        active_label_for_node = custom_node_label.strip() if custom_node_label.strip() else "Entity"
                    else:
                        active_label_for_node = node_label

                lbl_defaults = schema.get("default_properties", {}).get(node_label, {
                    "id_prefix": "node_", "name": "New Entity", "key1": "code", "val1": "E101", "key2": "status", "val2": "Active"
                })

                with col_id:
                    def_id_val = st.session_state.get("node_id_inp", f"{lbl_defaults.get('id_prefix', 'node_')}{len(graph.nodes) + 1}")
                    node_id_input = st.text_input("Node ID:", value=def_id_val, key="node_id_inp")

                def_name_val = st.session_state.get("node_name_inp", lbl_defaults.get("name", "New Entity"))
                node_name_input = st.text_input("Display Name:", value=def_name_val, key="node_name_inp")

                cp1, cp2 = st.columns(2)
                with cp1:
                    prop_k1 = st.text_input("Property 1 Key:", value=lbl_defaults.get("key1", "code"), key="prop_k1")
                    prop_v1 = st.text_input("Property 1 Val:", value=lbl_defaults.get("val1", "E101"), key="prop_v1")
                with cp2:
                    prop_k2 = st.text_input("Property 2 Key:", value=lbl_defaults.get("key2", "status"), key="prop_k2")
                    prop_v2 = st.text_input("Property 2 Val:", value=lbl_defaults.get("val2", "Active"), key="prop_v2")

                if st.button("Create Node", type="primary", use_container_width=True):
                    try:
                        props = {"name": node_name_input}
                        if prop_k1 and prop_v1:
                            try:
                                props[prop_k1] = float(prop_v1) if "." in prop_v1 else int(prop_v1)
                            except ValueError:
                                props[prop_k1] = prop_v1
                        if prop_k2 and prop_v2:
                            try:
                                props[prop_k2] = float(prop_v2) if "." in prop_v2 else int(prop_v2)
                            except ValueError:
                                props[prop_k2] = prop_v2

                        graph.add_node(node_id_input, [active_label_for_node], props)
                        st.session_state["matched_node_ids"] = [node_id_input]
                        st.session_state["trials"].append({
                            "Trial #": len(st.session_state["trials"]) + 1,
                            "Operation": "CREATE Node",
                            "Query / Action": f"CREATE (:{active_label_for_node} {{id: '{node_id_input}', name: '{node_name_input}'}})",
                            "Result": f"Node '{node_id_input}' added",
                            "Status": "Success",
                            "Timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        log_activity("Node Added", f"{node_id_input} (:{active_label_for_node})", "success")
                        st.toast(f"Node '{node_id_input}' created!")
                        for k in ["node_id_inp", "node_name_inp", "prop_v1", "prop_v2"]:
                            st.session_state.pop(k, None)
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Error: {str(ex)}")

            # SUB-TAB B: ADD RELATIONSHIP
            with sub_tab_rel:
                node_options = [f"{nid} ({':'.join(n.labels)}: {n.display_name()})" for nid, n in graph.nodes.items()]
                if len(node_options) < 2:
                    st.warning("Please create at least 2 nodes before creating a relationship.")
                else:
                    cs, ct = st.columns(2)
                    with cs:
                        src_choice = st.selectbox("From Node:", options=node_options, index=0)
                        src_id = src_choice.split(" ")[0]
                    with ct:
                        tgt_choice = st.selectbox("To Node:", options=node_options, index=min(1, len(node_options) - 1))
                        tgt_id = tgt_choice.split(" ")[0]

                    rel_options = schema["rel_types"]
                    def_rel_idx = 0
                    if "rel_type_sel" in st.session_state and st.session_state["rel_type_sel"] in rel_options:
                        def_rel_idx = rel_options.index(st.session_state["rel_type_sel"])
                    rel_type = st.selectbox("Relationship Type:", rel_options, index=def_rel_idx, key="rel_type_sel")
                    if rel_type == "Custom...":
                        custom_rel_val = st.session_state.get("custom_rel_inp", "CONNECTED_TO")
                        custom_rel_input = st.text_input("Custom Rel Type:", value=custom_rel_val, key="custom_rel_inp")
                        active_rel_type = custom_rel_input.strip().upper() if custom_rel_input.strip() else "CONNECTED_TO"
                    else:
                        active_rel_type = rel_type

                    rel_defaults = schema.get("default_rel_properties", {}).get(rel_type, {"key": "weight", "val": "1.0"})
                    crp1, crp2 = st.columns(2)
                    with crp1:
                        r_prop_k = st.text_input("Rel Prop Key:", value=rel_defaults.get("key", "weight"), key="r_prop_k")
                    with crp2:
                        r_prop_v = st.text_input("Rel Prop Val:", value=rel_defaults.get("val", "1.0"), key="r_prop_v")

                    if st.button("Create Relationship", type="primary", use_container_width=True):
                        try:
                            r_props = {}
                            if r_prop_k and r_prop_v:
                                try:
                                    r_props[r_prop_k] = float(r_prop_v) if "." in r_prop_v else int(r_prop_v)
                                except ValueError:
                                    r_props[r_prop_k] = r_prop_v
                            r = graph.add_relationship(src_id, tgt_id, active_rel_type, r_props)
                            st.session_state["matched_node_ids"] = [src_id, tgt_id]
                            st.session_state["matched_rel_ids"] = [r.id]
                            st.session_state["trials"].append({
                                "Trial #": len(st.session_state["trials"]) + 1,
                                "Operation": "CREATE Rel",
                                "Query / Action": f"CREATE ({src_id})-[:{active_rel_type}]->({tgt_id})",
                                "Result": f"Rel '{r.id}' added",
                                "Status": "Success",
                                "Timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            log_activity("Rel Added", f"({src_id})-[:{active_rel_type}]->({tgt_id})", "success")
                            st.toast(f"Relationship '{active_rel_type}' created!")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Error: {str(ex)}")

            # SUB-TAB C: SET PROPERTY
            with sub_tab_set:
                if not graph.nodes:
                    st.info("No nodes available.")
                else:
                    u_node_choice = st.selectbox("Target Node:", options=node_options, key="update_node_sel")
                    u_nid = u_node_choice.split(" ")[0]
                    selected_node = graph.nodes.get(u_nid)
                    if selected_node and selected_node.properties:
                        props_preview = " · ".join([f"**{k}**: {v}" for k, v in selected_node.properties.items()])
                        st.caption(f"Current: {props_preview}")

                    existing_keys = [k for k in selected_node.properties.keys() if k != "name"] if selected_node else []
                    prop_key_options = existing_keys + ["name", "Custom Key..."]
                    cu1, cu2 = st.columns(2)
                    with cu1:
                        prop_choice = st.selectbox("Property Key:", options=prop_key_options, key=f"pkc_{u_nid}")
                        if prop_choice == "Custom Key...":
                            set_k = st.text_input("Property Name:", value="status", key=f"csk_{u_nid}")
                        else:
                            set_k = prop_choice
                    with cu2:
                        current_val = str(selected_node.properties.get(set_k, "")) if selected_node else ""
                        set_v = st.text_input("New Value:", value=current_val, key=f"sv_{u_nid}_{set_k}")

                    if st.button("SET Property", type="primary", use_container_width=True):
                        try:
                            try:
                                v_parsed = float(set_v) if "." in set_v else int(set_v)
                            except ValueError:
                                v_parsed = set_v
                            succ, msg = graph.update_node(u_nid, {set_k: v_parsed})
                            st.session_state["matched_node_ids"] = [u_nid]
                            st.session_state["trials"].append({
                                "Trial #": len(st.session_state["trials"]) + 1,
                                "Operation": "SET Property",
                                "Query / Action": f"MATCH ({u_nid}) SET {set_k} = {set_v}",
                                "Result": msg[:40],
                                "Status": "Success" if succ else "Failed",
                                "Timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            log_activity("SET Prop", f"{u_nid}.{set_k} = {set_v}", "info")
                            st.toast(msg)
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Error: {str(ex)}")

            # SUB-TAB D: DELETE ENTITY
            with sub_tab_del:
                del_mode = st.radio("Delete Target:", ["Node", "Relationship"], horizontal=True)
                if del_mode == "Node":
                    del_node_choice = st.selectbox("Select Node:", options=node_options, key="del_n_choice")
                    del_id = del_node_choice.split(" ")[0] if del_node_choice else ""
                    detach_flag = st.checkbox("DETACH DELETE (remove connected rels)", value=True)
                else:
                    rel_options = [f"{rid} ({r.source} -[:{r.type}]-> {r.target})" for rid, r in graph.relationships.items()]
                    if not rel_options:
                        st.info("No relationships in graph.")
                        del_id = ""
                    else:
                        del_rel_choice = st.selectbox("Select Relationship:", options=rel_options, key="del_r_choice")
                        del_id = del_rel_choice.split(" ")[0]
                    detach_flag = False

                if st.button("Execute Delete", type="secondary", use_container_width=True):
                    if del_mode == "Node":
                        succ, msg = graph.delete_node(del_id, detach=detach_flag)
                    else:
                        succ, msg = graph.delete_relationship(del_id)

                    st.session_state["trials"].append({
                        "Trial #": len(st.session_state["trials"]) + 1,
                        "Operation": "DELETE",
                        "Query / Action": f"DETACH DELETE {del_id}" if detach_flag else f"DELETE {del_id}",
                        "Result": msg[:40],
                        "Status": "Success" if succ else "Failed",
                        "Timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    if succ:
                        st.session_state["matched_node_ids"] = []
                        st.session_state["matched_rel_ids"] = []
                        log_activity("Delete", f"{del_id} ({del_mode})", "warning")
                        st.toast(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_logger:
            col_lbtn1, col_lbtn2 = st.columns(2)
            with col_lbtn1:
                if st.button("Record Snapshot", type="primary", use_container_width=True):
                    trial_record = {
                        "Trial #": len(st.session_state["trials"]) + 1,
                        "Operation": "Graph Snapshot",
                        "Query / Action": f"Nodes: {metrics['num_nodes']}, Rels: {metrics['num_relationships']}",
                        "Result": f"Density: {metrics['density']}",
                        "Status": "Recorded",
                        "Timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state["trials"].append(trial_record)
                    log_activity("Snapshot", f"Nodes: {metrics['num_nodes']}", "info")
                    st.toast(f"Trial #{trial_record['Trial #']} logged!")
            with col_lbtn2:
                if st.button("Clear Trials", use_container_width=True):
                    st.session_state["trials"] = []
                    st.toast("Trials cleared.")

            if st.session_state["trials"]:
                df_trials = pd.DataFrame(st.session_state["trials"])
                st.dataframe(df_trials, use_container_width=True, hide_index=True, height=180)
                csv_data = df_trials.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="graph_db_trials.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.caption("No trials recorded yet. Click 'Record Snapshot' or perform graph operations.")

    with col_canvas:
        c_top1, c_top2 = st.columns([2.5, 1.2])
        with c_top1:
            view_mode = st.radio(
                "Engine:",
                ["Interactive Canvas", "Plotly Static"],
                horizontal=True,
                label_visibility="collapsed"
            )
        with c_top2:
            if st.button("Clear Highlight", use_container_width=True):
                st.session_state["matched_node_ids"] = []
                st.session_state["matched_rel_ids"] = []
                st.rerun()

        if view_mode == "Interactive Canvas":
            render_interactive_graph_canvas(
                graph=graph,
                matched_node_ids=st.session_state.get("matched_node_ids"),
                matched_rel_ids=st.session_state.get("matched_rel_ids")
            )
        else:
            fig = render_graph_figure(
                graph=graph,
                matched_node_ids=st.session_state.get("matched_node_ids"),
                matched_rel_ids=st.session_state.get("matched_rel_ids"),
                layout_algorithm="Spring (Force-Directed)",
                node_label_mode="Name / Label"
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def render_quiz_section():
    """Renders Section: Assessment Quiz in full scrollable layout with large, clear typography and instant feedback."""
    st.markdown(f"""
        <div class="purpose-header">
            <div class="hero-eyebrow"><span class="hero-dot"></span>PRACTICAL EVALUATION · {EXPERIMENT_CONFIG['lab_code']}</div>
            <h1>Knowledge Assessment Quiz</h1>
            <p class="subtitle">12 comprehensive questions evaluating your mastery of Property Graph concepts, Cypher patterns, and storage internals.</p>
        </div>
    """, unsafe_allow_html=True)

    is_submitted = st.session_state.get("quiz_submitted", False)
    if is_submitted:
        score = st.session_state.get("quiz_score", 0)
        total = len(QUIZ_QUESTIONS)
        perc = (score / total) * 100
        badge = "Distinction" if perc >= 80 else ("Passed" if perc >= 50 else "Needs Review")

        st.markdown(f"""
            <div class="quiz-score-banner">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                    <div>
                        <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); margin-bottom: 0.2rem;">Official Evaluation Result</div>
                        <span style="font-size: 2.2rem; font-weight: 800; color: var(--accent);">{score} / {total}</span>
                        <span style="font-size: 1.4rem; font-weight: 600; opacity: 0.85; margin-left: 0.5rem;">({perc:.0f}%)</span>
                        <div style="font-size: 1.18rem; font-weight: 700; margin-top: 0.3rem;">{badge}</div>
                    </div>
                    <div style="max-width: 450px;">
                        <p style="font-size: 1.05rem; line-height: 1.6; margin: 0; opacity: 0.9;">
                            Scroll down through each question to examine your answer choices, correct answers, and in-depth conceptual explanations.
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with st.form("graph_lab_quiz_form"):
        user_responses = {}

        for q in QUIZ_QUESTIONS:
            with st.container(border=True):
                st.markdown(f"""
                    <div class="quiz-q-badge">Question {q['id']:02d} of {len(QUIZ_QUESTIONS):02d}</div>
                    <div class="quiz-q-title">{q['question']}</div>
                """, unsafe_allow_html=True)

                saved_choice = st.session_state["quiz_answers"].get(q["id"], 0)
                selected = st.radio(
                    label=f"Options for Question {q['id']}:",
                    options=q["options"],
                    index=saved_choice if saved_choice < len(q["options"]) else 0,
                    key=f"quiz_radio_{q['id']}",
                    label_visibility="collapsed"
                )
                user_responses[q["id"]] = q["options"].index(selected)

                if is_submitted:
                    u_ans = st.session_state["quiz_answers"].get(q["id"])
                    correct_ans = q["answer_index"]
                    if u_ans == correct_ans:
                        st.success(f"**Correct.** You selected: **{q['options'][u_ans]}**")
                    else:
                        st.error(
                            f"**Incorrect.** Your selection: **{q['options'][u_ans]}** | "
                            f"**Correct answer:** **{q['options'][correct_ans]}**"
                        )
                    st.markdown(f"""
                        <div class="quiz-explanation-card">
                            <strong>Pedagogical Explanation:</strong> {q['explanation']}
                        </div>
                    """, unsafe_allow_html=True)

        st.write("")
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            btn_text = "Update & Re-evaluate Quiz Answers" if is_submitted else "Submit Quiz for Evaluation"
            submitted = st.form_submit_button(btn_text, type="primary", use_container_width=True)
        with col_btn2:
            reset_clicked = st.form_submit_button("Reset Answers", type="secondary", use_container_width=True)

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True
        for q in QUIZ_QUESTIONS:
            if user_responses.get(q["id"]) == q["answer_index"]:
                score += 1
        st.session_state["quiz_score"] = score
        log_activity("Quiz Done", f"Score: {score}/{len(QUIZ_QUESTIONS)}", "success")
        st.toast(f"Quiz evaluated! Final Score: {score}/{len(QUIZ_QUESTIONS)}")
        st.rerun()

    if reset_clicked:
        st.session_state["quiz_answers"] = {}
        st.session_state["quiz_submitted"] = False
        st.session_state["quiz_score"] = 0
        log_activity("Quiz Reset", "All quiz answers reset", "warning")
        st.toast("Quiz answers reset.")
        st.rerun()


def render_report_section():
    """Renders Section: Dynamic Lab Report Generator in a 2-column Single-Paper Layout with PDF Export."""
    st.markdown(f"""
        <div class="purpose-header" style="padding:1.2rem 1.8rem; margin-bottom:1.2rem;">
            <div class="hero-eyebrow"><span class="hero-dot"></span>PRACTICAL RECORD · {EXPERIMENT_CONFIG['lab_code']}</div>
            <h1 style="font-size:1.8rem !important; margin:0.2rem 0 !important;">Experiment Report Generation</h1>
            <p class="subtitle">Compile student details, observations, experimental benchmark records, and assessment score into an official PDF document.</p>
        </div>
    """, unsafe_allow_html=True)

    col_form, col_export = st.columns([1.0, 1.0])

    with col_form:
        with st.container(border=True):
            st.markdown('<div class="manual-label">Student Information</div>', unsafe_allow_html=True)
            student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                student_id = st.text_input("Roll / ID", value=st.session_state["student_info"].get("id", "21CS01"))
            with c_r2:
                lab_date = st.date_input("Date", value=datetime.now())

            st.session_state["student_info"]["name"] = student_name
            st.session_state["student_info"]["id"] = student_id
            st.session_state["student_info"]["date"] = str(lab_date)

            st.markdown('<div class="manual-label" style="margin-top:10px;">Observations & Conclusions</div>', unsafe_allow_html=True)
            student_notes = st.text_area(
                "Notes and inferences:",
                value=st.session_state.get("student_notes", (
                    "During the experiment, we successfully created, queried, and managed an interconnected property graph. "
                    "Using Cypher pattern matching, relationships were traversed efficiently without relational multi-table joins. "
                    "Referential constraints were verified when attempting plain DELETE on connected nodes, demonstrating the necessity "
                    "of DETACH DELETE for safely removing graph entities."
                )),
                height=130,
                label_visibility="collapsed"
            )
            st.session_state["student_notes"] = student_notes

    with col_export:
        with st.container(border=True):
            st.markdown('<div class="manual-label">Report Summary</div>', unsafe_allow_html=True)
            trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()
            graph_metrics = st.session_state["graph"].get_metrics()
            quiz_score = st.session_state.get("quiz_score", 0)
            quiz_total = len(QUIZ_QUESTIONS)
            
            st.markdown(
                f'<div class="report-facts">'
                f'<div class="report-fact"><small>Student</small><strong>{student_name}</strong></div>'
                f'<div class="report-fact"><small>Roll</small><strong>{student_id}</strong></div>'
                f'<div class="report-fact"><small>Nodes</small><strong>{graph_metrics["num_nodes"]}</strong></div>'
                f'<div class="report-fact"><small>Score</small><strong>{quiz_score} / {quiz_total}</strong></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Recorded Simulation Trials: **{len(trials_df)}** | Status: **{'Ready for Export' if quiz_score > 0 or len(trials_df) > 0 else 'Initial'}**")

            pdf_bytes = generate_pdf_report(
                student_name=student_name,
                student_id=student_id,
                date_str=str(lab_date),
                trials_df=trials_df,
                quiz_score=quiz_score,
                quiz_total=quiz_total,
                student_notes=student_notes,
                graph_metrics=graph_metrics
            )
            os.makedirs("static", exist_ok=True)
            with open("static/lab_report.pdf", "wb") as f:
                f.write(pdf_bytes)
            with open("lab_report.pdf", "wb") as f:
                f.write(pdf_bytes)

            st.write("")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.download_button(
                    label="Download lab_report.pdf",
                    data=pdf_bytes,
                    file_name=f"lab_report_{student_id}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            with col_b2:
                st.link_button(
                    "Open / Print PDF",
                    url="/app/static/lab_report.pdf",
                    use_container_width=True
                )


def render_certificate_section():
    """Renders Section: Certificate of Completion with verification details and PDF download."""
    st.markdown('<div class="manual-kicker">Credential & Accreditation</div>', unsafe_allow_html=True)
    st.header("Certificate of Completion")
    st.markdown('<div class="manual-intro">Verify your experiment participation and generate an official verifiable certificate of completion.</div>', unsafe_allow_html=True)

    col_meta1, col_meta2 = st.columns([1.1, 1.9])

    with col_meta1:
        with st.container(border=True):
            st.markdown('<div class="manual-label">Participant Details</div>', unsafe_allow_html=True)
            cert_name = st.text_input("Full Name", value=st.session_state["student_info"].get("name", "Student Name"), key="cert_name_input")
            cert_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "21CS01"), key="cert_id_input")
            cert_inst = st.text_input("Department / Institution", value=st.session_state.get("institution", "Department of Computer Science & Engineering"), key="cert_inst_input")
            cert_date = st.date_input("Issue Date", value=datetime.now(), key="cert_date_input")

            st.session_state["student_info"]["name"] = cert_name
            st.session_state["student_info"]["id"] = cert_id
            st.session_state["institution"] = cert_inst
            st.session_state["student_info"]["date"] = str(cert_date)

            st.divider()
            st.markdown('<div class="manual-label">Laboratory Criteria Status</div>', unsafe_allow_html=True)
            quiz_done = st.session_state.get("quiz_submitted", False)
            quiz_score = st.session_state.get("quiz_score", 0)
            quiz_total = len(QUIZ_QUESTIONS)
            num_trials = len(st.session_state.get("trials", []))

            if quiz_done:
                st.success(f"Assessment: Completed ({quiz_score} / {quiz_total} score)")
            else:
                st.info("Assessment: Pending (You can complete the Quiz in Section 'Quiz')")

            if num_trials > 0:
                st.success(f"Simulation Activity: {num_trials} trials recorded")
            else:
                st.caption(f"Simulation Activity: {num_trials} trials recorded (Try the 'Simulation' section)")

            # Generate unique verification hash / code
            unique_hash = hex(abs(hash(f"{cert_name}_{cert_id}_{cert_date}")))[2:10].upper()
            verification_id = f"VLAB-CS08-2026-{unique_hash}"
            st.caption(f"Verification Code: **{verification_id}**")

            # Generate PDF Certificate
            pdf_cert_bytes = generate_pdf_certificate(
                student_name=cert_name,
                student_id=cert_id,
                institution=cert_inst,
                date_str=str(cert_date),
                quiz_score=quiz_score,
                quiz_total=quiz_total,
                cert_id=verification_id
            )

            st.write("")
            st.download_button(
                label="Download Certificate (PDF)",
                data=pdf_cert_bytes,
                file_name=f"Certificate_{cert_id.replace(' ', '_')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

    with col_meta2:
        # Visual Certificate Preview Frame
        st.markdown(f"""
            <div class="certificate-preview-frame">
                <div class="cert-inner-frame">
                    <div class="cert-org">VIRTUAL LABORATORY  |  MINISTRY OF EDUCATION INITIATIVE</div>
                    <div class="cert-inst">{cert_inst.upper()}</div>
                    <div class="cert-title">CERTIFICATE OF COMPLETION</div>
                    <div class="cert-subtitle">This document is proudly awarded to</div>
                    <div class="cert-student">{cert_name}</div>
                    <div class="cert-id">Roll / Registration ID: <strong>{cert_id}</strong></div>
                    <div class="cert-body">
                        for successfully conducting the laboratory simulation, demonstrating competency in property graph modeling, 
                        mastering Cypher relationship queries, and completing the virtual lab requirements for:
                    </div>
                    <div class="cert-course">{EXPERIMENT_CONFIG['title'].upper()}</div>
                    <div class="cert-code">{EXPERIMENT_CONFIG['course']} | Course Code: {EXPERIMENT_CONFIG['lab_code']}</div>
                    <div class="cert-footer-row">
                        <div class="cert-col">
                            <div class="cert-meta-label">Date Issued</div>
                            <div class="cert-meta-val">{cert_date}</div>
                        </div>
                        <div class="cert-col cert-seal">
                            <div class="seal-badge">VERIFIED</div>
                            <div class="seal-sub">VLAB ACCREDITED</div>
                        </div>
                        <div class="cert-col">
                            <div class="cert-meta-label">Verification ID</div>
                            <div class="cert-meta-val" style="font-family:'IBM Plex Mono',monospace;">{verification_id}</div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_references_section():
    """Renders Section: References, Bibliography, Research Papers, and Standards."""
    st.markdown('<div class="manual-kicker">Academic Bibliography</div>', unsafe_allow_html=True)
    st.header("References & Further Reading")
    st.markdown('<div class="manual-intro">Curated academic literature, international standards, textbooks, and documentation on Graph Databases and Cypher.</div>', unsafe_allow_html=True)

    ref_tabs = st.tabs([
        "Core Textbooks", 
        "Seminal Research Papers", 
        "International Standards", 
        "Official Documentation", 
        "Libraries & Tools"
    ])

    with ref_tabs[0]:
        st.markdown("""
        ### Standard Textbooks & Monographs
        
        1. **Robinson, I., Webber, J., & Eifrem, E. (2015).**  
           *Graph Databases: New Opportunities for Connected Data* (2nd Edition). O'Reilly Media.  
           *Description:* The definitive practitioner reference written by the creators of Neo4j. Covers labeled property graph data modeling, query optimization, and architectural internals including native store files and Index-Free Adjacency.
        
        2. **Silberschatz, A., Korth, H. F., & Sudarshan, S. (2019).**  
           *Database System Concepts* (7th Edition). McGraw-Hill Education.  
           *Description:* Standard university database curriculum textbook. Features dedicated chapters on NoSQL paradigms, semi-structured data, and graph data architectures comparing relational joins to pointer-based graph traversals.
        
        3. **Needham, M., & Hodler, A. E. (2019).**  
           *Graph Algorithms: Practical Examples in Apache Spark and Neo4j*. O'Reilly Media.  
           *Description:* In-depth guide to pathfinding (Dijkstra, A*), centrality measures (PageRank, Betweenness), community detection (Louvain, Label Propagation), and similarity algorithms.
        
        4. **Easley, D., & Kleinberg, J. (2010).**  
           *Networks, Crowds, and Markets: Reasoning about a Highly Connected World*. Cambridge University Press.  
           *Description:* Foundational interdisciplinary text on graph theory, social network structure, information cascades, and power-law degree distributions.
        """)

    with ref_tabs[1]:
        st.markdown("""
        ### Seminal Research Papers
        
        1. **Euler, L. (1736).**  
           *"Solutio problematis ad geometriam situs pertinentis"* (*The solution of a problem relating to the geometry of position*).  
           *Commentarii Academiae Scientiarum Petropolitanae*, 8, 128–140.  
           *Significance:* The founding mathematical paper of graph theory and topology, resolving the historic Seven Bridges of Königsberg problem.
        
        2. **Francis, N., Green, A., Guagliardo, P., Libkin, L., Lindaaker, T., Marsault, V., Plantikow, S., Rydberg, M., Selmer, P., & Taylor, A. (2018).**  
           *"Cypher: An Open Cypher Query Language for Property Graphs"*.  
           *Proceedings of the 2018 International Conference on Management of Data (ACM SIGMOD)*, pp. 1809–1811.  
           *Significance:* Formalizes the operational semantics, pattern-matching mechanisms, and syntax grammar of the Cypher language.
        
        3. **Angles, R., & Gutierrez, C. (2008).**  
           *"Survey of Graph Database Models"*.  
           *ACM Computing Surveys (CSUR)*, 40(1), Article 1, 1–39.  
           *Significance:* Comprehensive survey detailing the evolution from 1960s network databases to modern hypergraphs, property graphs, and RDF triplestores.
        
        4. **Vicknair, C., Macias, M., Zhao, Z., Nan, X., Chen, Y., & Wilkins, D. (2010).**  
           *"A comparison of a graph database and a relational database: a data provenance perspective"*.  
           *Proceedings of the 48th Annual Southeast Regional Conference (ACM SE '10)*.  
           *Significance:* Empirical benchmark comparing relational multi-table joins against native graph traversal latency over increasing query hop depths.
        """)

    with ref_tabs[2]:
        st.markdown("""
        ### International Standards & Specifications
        
        1. **ISO/IEC 39075:2024 Information technology — Database languages — GQL**  
           *International Organization for Standardization (ISO) / International Electrotechnical Commission (IEC).*  
           *Published: April 2024.*  
           *Significance:* The official international standard for property graph databases. GQL is the first new full ISO standard database language since SQL was published in 1987.
        
        2. **The openCypher Project (v9 Specification)**  
           *openCypher Implementers Group.*  
           *Web:* [https://opencypher.org](https://opencypher.org)  
           *Significance:* The open industry standard specification enabling vendor-independent adoption of Cypher pattern matching across Neo4j, RedisGraph, Memgraph, AWS Neptune, and SAP HANA.
        
        3. **W3C Semantic Web Standards: RDF & SPARQL**  
           *World Wide Web Consortium (W3C).*  
           *Specifications:* RDF 1.1 Concepts and Abstract Syntax; SPARQL 1.1 Query Language.  
           *Web:* [https://www.w3.org/standards/semanticweb/](https://www.w3.org/standards/semanticweb/)  
           *Significance:* International standard for web-scale knowledge graphs and semantic triplestores using Subject-Predicate-Object semantics.
        """)

    with ref_tabs[3]:
        st.markdown("""
        ### Official Documentation & Developer Guides
        
        1. **Neo4j Official Cypher Manual**  
           *Neo4j Inc. Documentation Portal.*  
           *URL:* [https://neo4j.com/docs/cypher-manual/current/](https://neo4j.com/docs/cypher-manual/current/)  
           *Contents:* Complete clause reference (`MATCH`, `CREATE`, `MERGE`, `SET`, `DELETE`, `WITH`, `UNWIND`), pattern syntax, and aggregation functions.
        
        2. **Neo4j GraphAcademy**  
           *Interactive Certification & Learning Portal.*  
           *URL:* [https://graphacademy.neo4j.com/](https://graphacademy.neo4j.com/)  
           *Contents:* Professional pathways in Graph Data Modeling, Cypher Fundamentals, and Graph Data Science.
        
        3. **National Virtual Labs Project (Virtual Labs India)**  
           *Ministry of Education, Government of India / IIT Kharagpur.*  
           *URL:* [https://www.vlab.co.in/](https://www.vlab.co.in/)  
           *Contents:* Digital curriculum and simulation guidelines for undergraduate engineering laboratory courses.
        """)

    with ref_tabs[4]:
        st.markdown("""
        ### Python Libraries & Interactive Visualizers
        
        1. **NetworkX (Network Analysis in Python)**  
           *URL:* [https://networkx.org/](https://networkx.org/)  
           *Role:* High-productivity Python package for creating, manipulating, and studying the structure, dynamics, and functions of complex networks.
        
        2. **Plotly Graph Objects (Web Graph Visualization)**  
           *URL:* [https://plotly.com/python/](https://plotly.com/python/)  
           *Role:* Interactive HTML5 canvas and WebGL rendering engine used to render the dynamic 2D force-directed graph sandbox in this laboratory.
        
        3. **FPDF2 (Minimalist PDF Generation for Python)**  
           *URL:* [https://py-pdf.github.io/fpdf2/](https://py-pdf.github.io/fpdf2/)  
           *Role:* Document generation engine powering the laboratory report and official certificate export features.
        """)


# ======================================================================================
# 6. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "graph" not in st.session_state:
        g = PropertyGraph()
        g.load_university_graph()
        st.session_state["graph"] = g
    if "cypher_engine" not in st.session_state:
        st.session_state["cypher_engine"] = CypherEngine(st.session_state["graph"])
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "matched_node_ids" not in st.session_state:
        st.session_state["matched_node_ids"] = []
    if "matched_rel_ids" not in st.session_state:
        st.session_state["matched_rel_ids"] = []
    if "last_cypher_result" not in st.session_state:
        st.session_state["last_cypher_result"] = None
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "institution" not in st.session_state:
        st.session_state["institution"] = "Department of Computer Science & Engineering"
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "21CS01",
            "date": str(datetime.now().date()),
            "institution": "Department of Computer Science & Engineering"
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "preset_index" not in st.session_state:
        st.session_state["preset_index"] = 0
    if "active_preset" not in st.session_state:
        st.session_state["active_preset"] = "University Academic Knowledge Graph (Default)"
    if "activity_logs" not in st.session_state:
        st.session_state["activity_logs"] = [
            {"time": datetime.now().strftime("%H:%M:%S"), "action": "Ready", "type": "success", "detail": "University Graph loaded (11 nodes)"}
        ]


def log_activity(action: str, detail: str = "", log_type: str = "info"):
    """Appends an activity log entry to session state for the sidebar tracker."""
    if "activity_logs" not in st.session_state:
        st.session_state["activity_logs"] = []
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "detail": detail,
        "type": log_type
    }
    st.session_state["activity_logs"].append(entry)
    if len(st.session_state["activity_logs"]) > 40:
        st.session_state["activity_logs"] = st.session_state["activity_logs"][-40:]


def render_sidebar_logs():
    """Displays real-time user activity logs in the sidebar."""
    st.sidebar.divider()
    st.sidebar.markdown('<div class="manual-label" style="font-size:0.84rem; letter-spacing:0.1em; color:var(--accent); margin-bottom:0.4rem;">Live Activity Logs</div>', unsafe_allow_html=True)
    logs = st.session_state.get("activity_logs", [])
    if not logs:
        st.sidebar.caption("No activities recorded yet.")
        return

    color_map = {
        "success": ("#10b981", "rgba(16, 185, 129, 0.16)", "rgba(16, 185, 129, 0.45)"),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.16)", "rgba(245, 158, 11, 0.45)"),
        "error": ("#ef4444", "rgba(239, 68, 68, 0.16)", "rgba(239, 68, 68, 0.45)"),
        "query": ("#8b5cf6", "rgba(139, 92, 246, 0.16)", "rgba(139, 92, 246, 0.45)"),
        "info": ("#0284c7", "rgba(2, 132, 199, 0.16)", "rgba(2, 132, 199, 0.45)")
    }

    rows_html = []
    for entry in reversed(logs[-6:]):
        c_text, c_bg, c_border = color_map.get(entry["type"], color_map["info"])
        act = entry["action"]
        t_str = entry["time"]
        detail_txt = entry.get("detail", "")
        if len(detail_txt) > 28:
            detail_txt = detail_txt[:26] + "..."

        row = (
            '<div class="sidebar-log-row">'
            '<div class="sidebar-log-header">'
            f'<span class="sidebar-log-badge" style="background-color:{c_bg}; color:{c_text}; border:1px solid {c_border};">{act}</span>'
            f'<span class="sidebar-log-time">{t_str}</span>'
            '</div>'
            f'<span class="sidebar-log-detail">{detail_txt}</span>'
            '</div>'
        )
        rows_html.append(row)

    full_html = f'<div class="sidebar-logs-container">{"".join(rows_html)}</div>'
    st.sidebar.markdown(full_html, unsafe_allow_html=True)

    col_c1, col_c2 = st.sidebar.columns([1, 1])
    with col_c1:
        st.sidebar.caption(f"Total: {len(logs)}")
    with col_c2:
        if st.sidebar.button("Clear", key="clear_sidebar_logs_btn", use_container_width=True):
            st.session_state["activity_logs"] = []
            st.rerun()


def get_app_styles() -> str:
    """Generates clean, theme-adaptive CSS that renders seamlessly in both Light and Dark modes."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --accent: #0284c7;
        --accent-soft: rgba(2, 132, 199, 0.10);
        --accent-hover: #0369a1;
        --violet: #6366f1;
        --warm: #ea580c;
        --card-bg: rgba(127, 127, 127, 0.05);
        --card-border: rgba(127, 127, 127, 0.18);
        --card-subtle: rgba(127, 127, 127, 0.08);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --accent: #22d3c7;
            --accent-soft: rgba(34, 211, 199, 0.12);
            --accent-hover: #14b8a6;
            --violet: #7c6cf6;
            --warm: #f0a36a;
            --card-bg: rgba(255, 255, 255, 0.04);
            --card-border: rgba(255, 255, 255, 0.12);
            --card-subtle: rgba(255, 255, 255, 0.07);
        }
    }

    /* Container & Layout */
    .main .block-container {
        max-width: 1440px;
        padding: 0.8rem 2.2rem 2.2rem !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em !important;
    }
    
    h1 { font-size: 2.25rem !important; line-height: 1.15 !important; }
    h2 { font-size: 1.65rem !important; margin-top: 1.4rem !important; }
    h3 { font-size: 1.35rem !important; }
    h4 { font-size: 1.18rem !important; }

    /* Virtual Lab Header */
    .vlab-header {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-top: 3px solid var(--accent);
        border-radius: 12px;
        padding: 1.4rem 2rem 1.3rem;
        margin-bottom: 1.5rem;
    }
    .vlab-header h1 {
        margin: 0;
        font-size: 2.0rem !important;
        font-weight: 700 !important;
    }
    .hero-eyebrow { color: var(--accent); font-size: 0.84rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; }
    .hero-dot { display: inline-block; width: 8px; height: 8px; margin-right: 0.55rem; border-radius: 50%; background: var(--accent); vertical-align: 1px; }
    .hero-badge { background: linear-gradient(135deg, var(--accent), var(--violet)); color: #FFFFFF; padding: 0.32rem 0.72rem; border-radius: 999px; font-size: 0.84rem; font-weight: 700; letter-spacing: 0.08em; white-space: nowrap; }

    .manual-kicker, .manual-label {
        color: var(--accent) !important;
        font-size: 0.84rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.13em !important;
        text-transform: uppercase;
    }
    .manual-intro {
        border-left: 3px solid var(--accent);
        padding: 0.65rem 1.1rem;
        margin: 0.9rem 0 1.4rem;
        background: var(--accent-soft);
        border-radius: 0 8px 8px 0;
        font-size: 1.12rem !important;
        line-height: 1.7 !important;
    }

    .concept-diagram { display: flex; align-items: center; justify-content: center; gap: 1.2rem; padding: 1.2rem 1rem; margin: 0.5rem 0 1.5rem; border: 1px solid var(--card-border); background: var(--card-bg); border-radius: 10px; }
    .concept-diagram > div { display: grid; gap: 0.35rem; justify-items: center; }
    .concept-diagram small { opacity: 0.75; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .diagram-node { display: inline-flex; align-items: center; justify-content: center; width: 4.8rem; height: 2.8rem; border: 2px solid currentColor; background: var(--card-subtle); border-radius: 4px; font: 600 0.72rem 'IBM Plex Mono', monospace; }
    .diagram-node.accent { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
    .diagram-node.warm { border-color: var(--warm); color: var(--warm); background: rgba(234, 88, 12, 0.08); }
    .diagram-arrow { color: var(--accent); font-size: 1.5rem; }
    .diagram-arrow small { display: block; font-size: 0.6rem; text-align: center; opacity: 0.75; }

    .report-facts { display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--card-border); border-radius: 10px; background: var(--card-bg); margin: 0.8rem 0 1.2rem; }
    .report-fact { padding: 0.75rem 0.9rem; border-right: 1px solid var(--card-border); }
    .report-fact:last-child { border-right: 0; }
    .report-fact small { display: block; opacity: 0.75; font-size: 0.67rem; letter-spacing: 0.1em; text-transform: uppercase; }
    .report-fact strong { display: block; margin-top: 0.25rem; color: var(--accent); font-size: 1.05rem; overflow-wrap: anywhere; }

    .quiz-progress { height: 5px; background: var(--card-border); border-radius: 3px; overflow: hidden; margin: 0.8rem 0 1.4rem; }
    .quiz-progress span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--violet)); }

    /* Simulation metric chips */
    .metric-chips-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 3px; }
    .metric-chip { background: var(--card-subtle); border: 1px solid var(--card-border); border-radius: 6px; padding: 4px 9px; font-size: 0.85rem; }
    .metric-chip strong { color: var(--accent); font-weight: 700; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }
    .sidebar-title {
        color: var(--accent);
        font-size: 1.32rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .sidebar-code {
        opacity: 0.75;
        font-size: 0.76rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .sidebar-logs-container {
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin: 6px 0 10px;
    }
    .sidebar-log-row {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 6px;
        padding: 5px 8px;
    }
    .sidebar-log-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2px;
    }
    .sidebar-log-badge {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 1px 5px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .sidebar-log-time {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        opacity: 0.6;
    }
    .sidebar-log-detail {
        display: block;
        font-size: 0.76rem;
        opacity: 0.85;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-family: 'IBM Plex Mono', monospace;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-size: 1.18rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 0.95rem !important;
        border-radius: 8px !important;
        border: 1px solid var(--card-border) !important;
        background: var(--card-bg) !important;
        transition: all 0.15s ease-in-out !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: var(--card-subtle) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 1.18rem !important;
        font-weight: 600 !important;
        line-height: 1.3 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
        background: var(--accent-soft) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p {
        color: var(--accent) !important;
        font-weight: 700 !important;
    }

    /* General Typography & Spacing */
    body, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
        font-size: 1.12rem !important;
        line-height: 1.75 !important;
    }

    /* Form widgets, labels, inputs, selects, textareas */
    [data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] {
        font-size: 1.12rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stTextArea"] textarea,
    [data-testid="stDateInput"] input {
        font-size: 1.1rem !important;
    }

    /* All radio options (e.g. Quiz questions, presets) */
    [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.75rem;
    }
    [data-testid="stRadio"] div[role="radiogroup"] label {
        padding: 0.45rem 0.75rem !important;
        border-radius: 8px !important;
        transition: all 0.15s ease-in-out !important;
        cursor: pointer !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: var(--card-subtle) !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] label p,
    [data-testid="stRadio"] div[role="radiogroup"] label span {
        font-size: 1.12rem !important;
        line-height: 1.55 !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    button[kind="primary"], button[kind="secondary"], .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        font-size: 1.12rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.3rem !important;
        border-radius: 8px !important;
    }

    /* Captions & Alerts */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        font-size: 1.02rem !important;
        line-height: 1.55 !important;
        opacity: 0.88 !important;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        font-size: 1.12rem !important;
        line-height: 1.6 !important;
    }

    /* Quiz-specific Components */
    .quiz-score-banner {
        background: var(--card-bg);
        border: 2px solid var(--accent);
        border-radius: 14px;
        padding: 1.6rem 2.2rem;
        margin-bottom: 2rem;
    }
    .quiz-q-badge {
        display: inline-block;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--accent);
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .quiz-q-title {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        line-height: 1.6 !important;
        margin-bottom: 1rem !important;
    }
    .quiz-explanation-card {
        background: var(--card-subtle);
        border-left: 4px solid var(--accent);
        padding: 1rem 1.3rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        font-size: 1.08rem !important;
        line-height: 1.65 !important;
    }

    /* Purpose Section Cards */
    .purpose-header {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--accent);
        border-radius: 12px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .purpose-header h1 {
        font-size: 2.3rem !important;
        font-weight: 700 !important;
        margin: 0.4rem 0 !important;
    }
    .purpose-header p.subtitle {
        font-size: 1.12rem !important;
        opacity: 0.85;
        margin: 0;
    }
    .purpose-section-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .purpose-section-card h3 {
        font-size: 1.45rem !important;
        color: var(--accent) !important;
        margin-top: 0 !important;
        margin-bottom: 0.9rem !important;
    }
    .purpose-section-card p, .purpose-section-card li {
        font-size: 1.12rem !important;
        line-height: 1.7 !important;
    }
    .highlight-box {
        background: var(--accent-soft);
        border-left: 4px solid var(--accent);
        padding: 1rem 1.3rem;
        border-radius: 0 8px 8px 0;
        margin: 1.2rem 0;
        font-size: 1.12rem !important;
    }
    .vs-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.3rem;
        margin: 1.3rem 0;
    }
    @media (max-width: 800px) {
        .vs-grid { grid-template-columns: 1fr; }
    }
    .vs-box {
        background: var(--card-subtle);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
    }
    .vs-box.bad {
        border-left: 4px solid #ef4444;
    }
    .vs-box.good {
        border-left: 4px solid #10b981;
    }
    .timeline-card {
        border-left: 3px solid var(--accent);
        padding: 0.6rem 0 0.8rem 1.4rem;
        margin-bottom: 1.1rem;
        position: relative;
    }
    .timeline-card::before {
        content: '';
        position: absolute;
        left: -8px;
        top: 0.8rem;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: var(--accent);
    }
    .timeline-year {
        font-weight: 800;
        color: var(--accent);
        font-size: 1.2rem;
    }
    .timeline-title {
        font-weight: 700;
        font-size: 1.12rem;
        margin: 0.15rem 0 0.4rem;
    }
    .usecase-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.1rem;
        margin-top: 1.2rem;
    }
    .usecase-card {
        background: var(--card-subtle);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .usecase-card:hover {
        border-color: var(--accent);
        transform: translateY(-2px);
    }
    .usecase-card .icon {
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
    }
    .usecase-card strong {
        display: block;
        font-size: 1.15rem;
        color: var(--accent);
        margin-bottom: 0.4rem;
    }
    .usecase-card p {
        font-size: 1.12rem !important;
        line-height: 1.6 !important;
        margin: 0;
        opacity: 0.9;
    }

    /* Certificate Styling */
    .certificate-preview-frame {
        background: #ffffff;
        color: #0f172a;
        border: 4px solid #1e3a8a;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        margin-top: 0.5rem;
    }
    .cert-inner-frame {
        border: 2px solid #d97706;
        border-radius: 8px;
        padding: 2rem 1.8rem;
        text-align: center;
        background: linear-gradient(180deg, #fafafa 0%, #ffffff 100%);
    }
    .cert-org {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #64748b;
        text-transform: uppercase;
    }
    .cert-inst {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-top: 0.3rem;
        letter-spacing: 0.05em;
    }
    .cert-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
        margin: 1.1rem 0 0.4rem;
        letter-spacing: 0.04em;
        font-family: 'Space Grotesk', sans-serif;
    }
    .cert-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        font-style: italic;
    }
    .cert-student {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0284c7;
        margin: 0.8rem 0 0.2rem;
        text-decoration: underline;
        text-decoration-color: #d97706;
        text-underline-offset: 6px;
    }
    .cert-id {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 0.9rem;
    }
    .cert-body {
        font-size: 0.95rem;
        color: #334155;
        line-height: 1.6;
        max-width: 580px;
        margin: 0 auto 0.9rem;
    }
    .cert-course {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .cert-code {
        font-size: 0.85rem;
        color: #2563eb;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .cert-footer-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #e2e8f0;
        padding-top: 1rem;
        margin-top: 1rem;
    }
    .cert-col {
        flex: 1;
        text-align: center;
    }
    .cert-meta-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .cert-meta-val {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.15rem;
    }
    .seal-badge {
        display: inline-block;
        background: #fef3c7;
        color: #b45309;
        border: 1px solid #d97706;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    .seal-sub {
        font-size: 0.65rem;
        color: #92400e;
        margin-top: 0.2rem;
        font-weight: 700;
    }

    /* Tabs & Code Areas */
    [data-testid="stTabs"] [role="tablist"] { gap: 0.5rem; border-bottom: 1px solid var(--card-border); }
    [data-testid="stTabs"] button[role="tab"] { border-radius: 8px 8px 0 0; padding: 0.9rem 1.4rem; font-size: 1.35rem !important; font-weight: 600 !important; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--accent); font-weight: 700; }
    [data-testid="stTextArea"] textarea { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.05rem !important; }
    </style>
    """


def main():
    st.set_page_config(
        page_title="Create and Manage a Graph Database - Virtual Lab",
        page_icon=None,
        layout="wide"
    )

    init_session_state()

    # Dynamic application theme
    st.markdown(get_app_styles(), unsafe_allow_html=True)

    # Navigation Sidebar
    st.sidebar.markdown(f'<div class="sidebar-title">Graph Database Lab</div>', unsafe_allow_html=True)

    navigation_options = [
        "Purpose",
        "Theory",
        "Simulation",
        "Quiz",
        "Report Generation",
        "Certificate",
        "References"
    ]
    requested_section = st.session_state.get("requested_section", navigation_options[0])
    if requested_section not in navigation_options:
        requested_section = navigation_options[0]

    section = st.sidebar.radio(
        "Navigation",
        options=navigation_options,
        index=navigation_options.index(requested_section),
        label_visibility="collapsed"
    )
    st.session_state["requested_section"] = section

    # Activity Logs in Sidebar
    render_sidebar_logs()

    # Section Dispatcher
    if section == "Purpose":
        render_purpose_section()
    elif section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()
    elif section == "Certificate":
        render_certificate_section()
    elif section == "References":
        render_references_section()


if __name__ == "__main__":
    main()




