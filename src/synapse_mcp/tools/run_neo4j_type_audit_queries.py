import os

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "synapse")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

queries = [
    (
        "MATCH (e:Entity {name: 'Download'}) RETURN DISTINCT e.type AS type",
        "Download types",
    ),
    (
        "MATCH (e:Entity {name: 'Claude Code'}) RETURN DISTINCT e.type AS type",
        "Claude Code types",
    ),
    (
        "MATCH (e:Entity {name: 'F'}) WHERE e.type = 'Product' RETURN count(e) AS count",
        "F as Product count",
    ),
    (
        "MATCH (e:Entity) WHERE size(e.name) <= 2 AND e.type = 'Product' AND NOT e.name IN ['AI', 'ML', 'LLM', 'GPT', 'NLP', 'AGI', 'Q&A', 'CoT', 'MoE', 'I/O', 'OS', 'UI', 'UX', 'CPU', 'GPU'] AND NOT e.name =~ '^[0-9]+$' RETURN count(e) AS count",
        "Single-char/minimal Product entities",
    ),
    (
        "MATCH (e:Entity) WHERE (e.name CONTAINS '<' OR e.name CONTAINS '>') AND (e.type = 'Person' OR e.type = 'Organization') RETURN count(e) AS count",
        "HTML/XML as Person/Org",
    ),
    (
        "MATCH (e:Entity) WHERE e.name STARTS WITH 'http' AND (e.type = 'Person' OR e.type = 'Location' OR e.type = 'Organization') RETURN count(e) AS count",
        "URL as Person/Loc/Org",
    ),
    (
        "MATCH (e:Entity) WHERE size(e.name) <= 2 AND NOT e.name IN ['AI', 'ML', 'LLM', 'GPT', 'NLP', 'AGI', 'Q&A', 'CoT', 'MoE', 'I/O', 'OS', 'UI', 'UX', 'CPU', 'GPU'] AND NOT e.name =~ '^[0-9]+$' AND (e.type = 'Person' OR e.type = 'Organization') RETURN count(e) AS count",
        "Single-char/minimal as Person/Org",
    ),
    (
        "MATCH (e:Entity {type: 'Entity'}) RETURN count(e) AS count_generic_entity",
        "Generic 'Entity' count",
    ),
]

results: list[tuple[str, list[str] | int | str]] = []
with driver.session(database=NEO4J_DATABASE) as session:
    for query_str, query_name in queries:
        try:
            if "RETURN DISTINCT" in query_str:
                result = session.run(query_str)
                types = [record["type"] for record in result]
                results.append((query_name, types))
            else:
                result = session.run(query_str).single()
                count = result.value() if result else 0
                results.append((query_name, count))
        except Exception as e:
            results.append((query_name, f"Error: {e}"))

driver.close()

for query_name, result in results:
    print(f"{query_name}: {result}")
