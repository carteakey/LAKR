import os
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from tabulate import tabulate


class Neo4jGraphStats:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def close(self):
        self.driver.close()

    def run_query(self, query):
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return result.single()
        except Neo4jError as e:
            self.logger.error(f"Neo4j query failed: {query}\nError: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error during query execution: {query}\nError: {e}"
            )
            return None

    def get_node_count(self):
        self.logger.info("Fetching node count...")
        result = self.run_query("MATCH (n) RETURN count(n) AS nodeCount")
        return result.get("nodeCount") if result else 0

    def get_relationship_count(self):
        self.logger.info("Fetching relationship count...")
        result = self.run_query("MATCH ()-->() RETURN count(*) AS relationshipCount")
        return result.get("relationshipCount") if result else 0

    def get_nodes_by_label(self):
        self.logger.info("Fetching nodes by label...")
        query = """
        CALL db.labels() YIELD label
        CALL {
            WITH label
            MATCH (n)
            WHERE label in labels(n)
            RETURN count(n) AS count
        }
        RETURN label, count
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {record["label"]: record["count"] for record in result}

    def get_relationships_by_type(self):
        self.logger.info("Fetching relationships by type...")
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        CALL {
            WITH relationshipType
            MATCH ()-[r]->()
            WHERE type(r) = relationshipType
            RETURN count(r) AS count
        }
        RETURN relationshipType, count
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {record["relationshipType"]: record["count"] for record in result}

    def get_density(self):
        self.logger.info("Calculating graph density...")
        node_count = self.get_node_count()
        relationship_count = self.get_relationship_count()
        max_edges = node_count * (node_count - 1) / 2.0
        return relationship_count / max_edges if max_edges > 0 else 0

    def get_clustering_coefficient(self):
        self.logger.info("Calculating average clustering coefficient...")
        query = """
        MATCH (n)
        WHERE (n)--() 
        WITH n, size((n)--()) as degree
        WHERE degree > 1
        WITH n, degree
        LIMIT 10000
        MATCH (n)--()-[r]-()--(n)
        WITH n, degree, count(DISTINCT r) AS triangles
        RETURN avg(2.0 * triangles / (degree * (degree - 1))) AS avgClusteringCoefficient
        """
        result = self.run_query(query)
        return result.get("avgClusteringCoefficient") if result else 0

    def get_node_property_completeness(self):
        self.logger.info("Calculating average node property completeness...")
        query = """
        MATCH (n)
        RETURN avg(size(keys(n))) AS avgNodeProperties
        """
        result = self.run_query(query)
        return result.get("avgNodeProperties") if result else 0

    def get_relationship_property_completeness(self):
        self.logger.info("Calculating average relationship property completeness...")
        query = """
        MATCH ()-[r]->()
        RETURN avg(size(keys(r))) AS avgRelationshipProperties
        """
        result = self.run_query(query)
        return result.get("avgRelationshipProperties") if result else 0

    def get_isolated_nodes(self):
        self.logger.info("Counting isolated nodes...")
        query = """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN count(n) AS isolatedNodeCount
        """
        result = self.run_query(query)
        return result.get("isolatedNodeCount") if result else 0

    def get_degree_distribution(self):
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        RETURN min(degree) AS minDegree,
               max(degree) AS maxDegree,
               avg(degree) AS avgDegree,
               percentileCont(degree, 0.5) AS medianDegree
        """
        result = self.run_query(query)
        return (
            result
            if result
            else {"minDegree": 0, "maxDegree": 0, "avgDegree": 0, "medianDegree": 0}
        )

    def get_stats_latex(self):
        stats = self.get_stats()

        latex_table = (
            "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|r|}\n\\hline\n"
        )
        latex_table += "\\textbf{Metric} & \\textbf{Value} \\\\ \\hline\n"

        for metric, value in stats:
            if metric == "":
                latex_table += "\\hline\n"
            elif metric.startswith("  "):
                # This is a sub-item (like individual labels or relationship types)
                latex_table += f"\\quad {metric.strip()} & {value} \\\\ \n"
            else:
                latex_table += f"{metric} & {value} \\\\ \n"

        latex_table += "\\hline\n\\end{tabular}\n"
        latex_table += "\\caption{Neo4j Graph Statistics}\n"
        latex_table += "\\label{tab:neo4j-stats}\n"
        latex_table += "\\end{table}"

        return latex_table

    def get_stats(self):
        self.logger.info("Fetching all statistics...")
        node_count = self.get_node_count()
        relationship_count = self.get_relationship_count()
        nodes_by_label = self.get_nodes_by_label()
        relationships_by_type = self.get_relationships_by_type()
        density = self.get_density()
        clustering_coefficient = self.get_clustering_coefficient()
        node_property_completeness = self.get_node_property_completeness()
        relationship_property_completeness = (
            self.get_relationship_property_completeness()
        )
        isolated_nodes = self.get_isolated_nodes()
        degree_distribution = self.get_degree_distribution()

        stats = [
            ["Total Nodes", node_count],
            ["Total Relationships", relationship_count],
            ["Graph Density", round(density, 6)],
            ["Avg Clustering Coefficient", round(clustering_coefficient, 4)],
            ["Avg Node Properties", round(node_property_completeness, 2)],
            [
                "Avg Relationship Properties",
                round(relationship_property_completeness, 2),
            ],
            ["Isolated Nodes", isolated_nodes],
            ["Min Degree", degree_distribution["minDegree"]],
            ["Max Degree", degree_distribution["maxDegree"]],
            ["Avg Degree", round(degree_distribution["avgDegree"], 2)],
            ["Median Degree", degree_distribution["medianDegree"]],
        ]

        stats.append(["", ""])
        stats.append(["Nodes by Label", ""])
        for label, count in nodes_by_label.items():
            stats.append([f"  {label}", count])

        stats.append(["", ""])
        stats.append(["Relationships by Type", ""])
        for rel_type, count in relationships_by_type.items():
            stats.append([f"  {rel_type}", count])

        return stats


def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "tmu-2024")

    stats = Neo4jGraphStats(uri, user, password)
    table = stats.get_stats()
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

    latex_table = stats.get_stats_latex()
    # print(latex_table)

    stats.close()


if __name__ == "__main__":
    main()
