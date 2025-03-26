import lancedb
import numpy as np

def search(embedding: list):
    # Connect to LanceDB directory
    db = lancedb.connect("lancedb_dir")

    # Open the existing table
    table_name = "ghazals"
    table = db.open_table(name=table_name)

    # Ensure the embedding is a Python list, not a NumPy array
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    print(table.count_rows())
    # Perform vector-based search       
    results = table.search(embedding, vector_column_name="embedding") \
                   .select(["id", "document"]) \
                   .to_pandas()

    return results
