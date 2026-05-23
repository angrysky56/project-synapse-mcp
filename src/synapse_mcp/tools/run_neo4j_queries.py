import os

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "synapse")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

queries = [
    "MATCH (e:Entity) WHERE e.name CONTAINS '<' OR e.name CONTAINS '>' RETURN count(e) AS count",
    "MATCH (e:Entity) WHERE e.name STARTS WITH 'http' RETURN count(e) AS count",
    "MATCH (e:Entity) WHERE e.name IN ['Fig','Figure','Table','Section','Appendix','Download','Refer','Note','See','Ref','References','et al','et al.','Ibid'] RETURN count(e) AS count",
    "MATCH (e:Entity) WHERE size(e.name) <= 2 AND NOT e.name IN ['AI', 'ML', 'LLM', 'GPT', 'NLP', 'AGI', 'Q&A', 'CoT', 'MoE', 'I/O', 'OS', 'UI', 'UX', 'CPU', 'GPU'] AND NOT e.name =~ '^[0-9]+$' RETURN count(e) AS count",
    "MATCH (e:Entity) WHERE e.name CONTAINS '\\\\' OR e.name CONTAINS '\\\\math' OR e.name CONTAINS '\\\\times' OR e.name CONTAINS '\\\\sum' OR e.name CONTAINS '\\\\int' RETURN count(e) AS count",
    "MATCH (e:Entity) WHERE e.name =~ '^[0-9]+$' AND e.type IN ['Person', 'Organization', 'Location'] RETURN count(e) AS count",
    "MATCH (e:Entity) WHERE e.name CONTAINS '/' AND NOT e.name STARTS WITH 'http' AND NOT e.name CONTAINS '.' RETURN count(e) AS count",
]

results = []
with driver.session(database=NEO4J_DATABASE) as session:
    for query in queries:
        try:
            result = session.run(query).single()["count"]
            results.append(result)
        except Exception as e:
            results.append(f"Error: {e}")

driver.close()

for i, count in enumerate(results):
    print(f"Query {i+1} result: {count}")
