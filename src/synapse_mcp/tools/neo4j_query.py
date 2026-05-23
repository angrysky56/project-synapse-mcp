import os

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")  # Attempting with empty password

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def get_entity_type_counts(tx):
    query = "MATCH (e:Entity) RETURN e.type, count(e) AS count ORDER BY count DESC"
    result = tx.run(query)
    return [{"type": record["e.type"], "count": record["count"]} for record in result]


with driver.session() as session:
    entity_type_counts = session.execute_read(get_entity_type_counts)
    for item in entity_type_counts:
        print(f"Type: {item['type']}, Count: {item['count']}")

driver.close()
