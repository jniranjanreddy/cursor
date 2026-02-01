"""
Database CRUD Practice: MongoDB, PostgreSQL, Milvus (vector DB)
Basic to Advanced operations for all three.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Optional imports (install: pip install pymongo psycopg2-binary pymilvus python-dotenv)
# -----------------------------------------------------------------------------
import pymongo
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from bson.objectid import ObjectId

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from pymilvus import MilvusClient
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False

# -----------------------------------------------------------------------------
# MongoDB connection
# -----------------------------------------------------------------------------
LOCAL_MONGO_URI = os.getenv("LOCAL_MONGO_URI")
mongo_client = MongoClient(LOCAL_MONGO_URI)
db = mongo_client["testdb"]
collection = db["testcollection"]

# -----------------------------------------------------------------------------
# PostgreSQL connection (set POSTGRES_URI or POSTGRES_HOST, POSTGRES_PORT, etc.)
# -----------------------------------------------------------------------------
POSTGRES_URI = os.getenv("POSTGRES_URI")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "testdb")

def get_pg_conn():
    """Get PostgreSQL connection."""
    if not PSYCOPG2_AVAILABLE:
        raise ImportError("pip install psycopg2-binary")
    if POSTGRES_URI:
        return psycopg2.connect(POSTGRES_URI)
    return psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT,
        user=POSTGRES_USER, password=POSTGRES_PASSWORD, dbname=POSTGRES_DB
    )

# -----------------------------------------------------------------------------
# Milvus connection (set MILVUS_URI e.g. http://localhost:19530)
# -----------------------------------------------------------------------------
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "practise_vectors")

def get_milvus_client():
    """Get Milvus client (requires pymilvus)."""
    if not PYMILVUS_AVAILABLE:
        raise ImportError("pip install pymilvus")
    return MilvusClient(uri=MILVUS_URI)


# =============================================================================
# BASIC CRUD
# =============================================================================

def basic_create():
    """Create: insert_one and insert_many"""
    # Insert single document
    doc = {"name": "John Doe", "age": 30, "email": "john.doe@example.com"}
    result = collection.insert_one(doc)
    print(f"Inserted one: {result.inserted_id}")

    # Insert many documents
    docs = [
        {"name": "Jane Smith", "age": 25, "email": "jane@example.com"},
        {"name": "Bob Wilson", "age": 35, "email": "bob@example.com"},
    ]
    result = collection.insert_many(docs)
    print(f"Inserted many: {result.inserted_ids}")


def basic_read():
    """Read: find_one and find"""
    # Find one document
    doc = collection.find_one({"name": "John Doe"})
    print("Find one:", doc)

    # Find all matching
    cursor = collection.find({"age": {"$gte": 25}})
    for d in cursor:
        print(d)


def basic_update():
    """Update: update_one and update_many"""
    # Update one
    result = collection.update_one(
        {"name": "John Doe"},
        {"$set": {"age": 31, "updated": True}}
    )
    print(f"Modified: {result.modified_count}")

    # Update many
    result = collection.update_many(
        {"age": {"$lt": 40}},
        {"$set": {"category": "young"}}
    )
    print(f"Modified: {result.modified_count}")


def basic_delete():
    """Delete: delete_one and delete_many"""
    # Delete one
    result = collection.delete_one({"name": "Bob Wilson"})
    print(f"Deleted: {result.deleted_count}")

    # Delete many
    result = collection.delete_many({"category": "young"})
    print(f"Deleted: {result.deleted_count}")


# =============================================================================
# INTERMEDIATE CRUD
# =============================================================================

def intermediate_read():
    """Read with projection, sort, limit, skip"""
    # Projection: only return certain fields (1 = include, 0 = exclude)
    cursor = collection.find(
        {"age": {"$exists": True}},
        {"name": 1, "email": 1, "_id": 0}
    )
    for d in cursor:
        print(d)

    # Sort (1 = asc, -1 = desc), limit, skip
    cursor = collection.find().sort("age", -1).limit(5).skip(0)
    for d in cursor:
        print(d)


def intermediate_update():
    """Update with $inc, $push, $unset, upsert"""
    # $inc: increment numeric field
    collection.update_one(
        {"name": "John Doe"},
        {"$inc": {"age": 1, "login_count": 1}}
    )

    # $push: add to array
    collection.update_one(
        {"name": "John Doe"},
        {"$push": {"tags": "vip"}}
    )

    # $unset: remove field
    collection.update_one(
        {"name": "John Doe"},
        {"$unset": {"updated": ""}}
    )

    # Upsert: insert if no match
    collection.update_one(
        {"email": "new@example.com"},
        {"$set": {"name": "New User", "age": 20}},
        upsert=True
    )


def intermediate_delete():
    """Delete with conditions"""
    # Delete by ObjectId
    oid = ObjectId("your_document_id_here")
    collection.delete_one({"_id": oid})

    # Delete all documents in collection (use with caution)
    # result = collection.delete_many({})


# =============================================================================
# ADVANCED CRUD
# =============================================================================

def advanced_read():
    """Complex queries: $and, $or, $in, $regex, $elemMatch"""
    # $and (implicit when multiple keys in same dict)
    cursor = collection.find({
        "age": {"$gte": 25, "$lte": 40},
        "name": {"$regex": "^J", "$options": "i"}
    })

    # $or
    cursor = collection.find({
        "$or": [
            {"name": "John Doe"},
            {"email": "jane@example.com"}
        ]
    })

    # $in
    cursor = collection.find({"age": {"$in": [25, 30, 35]}})

    # $elemMatch (for arrays)
    cursor = collection.find({
        "tags": {"$elemMatch": {"$eq": "vip"}}
    })

    # Combined
    cursor = collection.find({
        "$and": [
            {"age": {"$gte": 20}},
            {"$or": [{"category": "young"}, {"category": {"$exists": False}}]}
        ]
    })
    return list(cursor)


def advanced_update():
    """Update with array operators and pipeline-style updates"""
    # $addToSet: add to array only if not present
    collection.update_one(
        {"name": "John Doe"},
        {"$addToSet": {"tags": "premium"}}
    )

    # $pull: remove from array
    collection.update_one(
        {"name": "John Doe"},
        {"$pull": {"tags": "vip"}}
    )

    # $set with dot notation (nested fields)
    collection.update_one(
        {"name": "John Doe"},
        {"$set": {"address.city": "NYC", "address.zip": "10001"}}
    )

    # Update with aggregation pipeline (MongoDB 4.2+)
    collection.update_one(
        {"name": "John Doe"},
        [
            {"$set": {"age": {"$add": ["$age", 1]}, "last_modified": "$$NOW"}}
        ]
    )


def advanced_delete():
    """Delete with complex filters"""
    # Delete documents matching complex query
    result = collection.delete_many({
        "age": {"$lt": 18},
        "status": "inactive"
    })
    print(f"Deleted: {result.deleted_count}")


def bulk_write():
    """Advanced: Bulk write (mixed insert/update/delete in one round-trip)"""
    from pymongo import InsertOne, UpdateOne, DeleteOne

    operations = [
        InsertOne({"name": "Bulk User 1", "age": 22}),
        UpdateOne({"name": "John Doe"}, {"$set": {"bulk_updated": True}}),
        DeleteOne({"name": "Bulk User 1"}),
    ]
    try:
        result = collection.bulk_write(operations)
        print(f"Inserted: {result.inserted_count}, Modified: {result.modified_count}, Deleted: {result.deleted_count}")
    except BulkWriteError as e:
        print(f"Bulk write error: {e.details}")


def aggregation_example():
    """Advanced: Aggregation pipeline"""
    pipeline = [
        {"$match": {"age": {"$gte": 20}}},
        {"$group": {"_id": "$category", "count": {"$sum": 1}, "avg_age": {"$avg": "$age"}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]
    for doc in collection.aggregate(pipeline):
        print(doc)


def find_one_and_operations():
    """Advanced: findOneAndUpdate, findOneAndReplace, findOneAndDelete"""
    # findOneAndUpdate: atomically update and return
    doc = collection.find_one_and_update(
        {"name": "John Doe"},
        {"$set": {"last_login": "2025-01-30"}},
        return_document=pymongo.ReturnDocument.AFTER
    )
    print("Updated doc:", doc)

    # findOneAndReplace
    doc = collection.find_one_and_replace(
        {"name": "Jane Smith"},
        {"name": "Jane Smith", "age": 26, "email": "jane.new@example.com"},
        return_document=pymongo.ReturnDocument.AFTER
    )

    # findOneAndDelete
    doc = collection.find_one_and_delete({"name": "Bob Wilson"})
    print("Deleted doc:", doc)


# =============================================================================
# POSTGRESQL CRUD (basic to advanced)
# =============================================================================

def pg_ensure_table(conn):
    """Create test table if not exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                age INTEGER,
                email VARCHAR(255) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
        """)
        conn.commit()


def pg_basic_create():
    """Postgres: INSERT one and many."""
    if not PSYCOPG2_AVAILABLE:
        print("pip install psycopg2-binary")
        return
    conn = get_pg_conn()
    pg_ensure_table(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (name, age, email) VALUES (%s, %s, %s) RETURNING id",
                ("John Doe", 30, "john.doe@example.com")
            )
            row_id = cur.fetchone()[0]
            print(f"Inserted one, id: {row_id}")
            cur.executemany(
                "INSERT INTO users (name, age, email) VALUES (%s, %s, %s)",
                [("Jane Smith", 25, "jane@example.com"), ("Bob Wilson", 35, "bob@example.com")]
            )
            print(f"Inserted many: {cur.rowcount}")
        conn.commit()
    finally:
        conn.close()


def pg_basic_read():
    """Postgres: SELECT one and many."""
    if not PSYCOPG2_AVAILABLE:
        return
    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE name = %s", ("John Doe",))
            row = cur.fetchone()
            print("Find one:", dict(row) if row else None)
            cur.execute("SELECT * FROM users WHERE age >= %s", (25,))
            for row in cur.fetchall():
                print(dict(row))
    finally:
        conn.close()


def pg_basic_update():
    """Postgres: UPDATE one and many."""
    if not PSYCOPG2_AVAILABLE:
        return
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET age = %s, metadata = %s WHERE name = %s",
                (31, '{"updated": true}', "John Doe")
            )
            print(f"Updated one: {cur.rowcount}")
            cur.execute("UPDATE users SET metadata = %s WHERE age < %s", ('{"category": "young"}', 40))
            print(f"Updated many: {cur.rowcount}")
        conn.commit()
    finally:
        conn.close()


def pg_basic_delete():
    """Postgres: DELETE one and many."""
    if not PSYCOPG2_AVAILABLE:
        return
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE name = %s", ("Bob Wilson",))
            print(f"Deleted: {cur.rowcount}")
            cur.execute("DELETE FROM users WHERE metadata->>'category' = %s", ("young",))
            print(f"Deleted many: {cur.rowcount}")
        conn.commit()
    finally:
        conn.close()


def pg_intermediate_read():
    """Postgres: SELECT with ORDER BY, LIMIT, OFFSET, specific columns."""
    if not PSYCOPG2_AVAILABLE:
        return
    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT name, email FROM users WHERE age IS NOT NULL ORDER BY age DESC LIMIT 5 OFFSET 0"
            )
            for row in cur.fetchall():
                print(dict(row))
    finally:
        conn.close()


def pg_intermediate_update():
    """Postgres: UPDATE with RETURNING, CASE, COALESCE."""
    if not PSYCOPG2_AVAILABLE:
        return
    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                UPDATE users SET age = age + 1
                WHERE name = %s RETURNING id, name, age
            """, ("John Doe",))
            row = cur.fetchone()
            if row:
                print("Updated and returned:", dict(row))
        conn.commit()
    finally:
        conn.close()


def pg_advanced_read():
    """Postgres: JOINs, subqueries, CTE, JSONB."""
    if not PSYCOPG2_AVAILABLE:
        return
    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # CTE and aggregation
            cur.execute("""
                WITH age_groups AS (
                    SELECT (age / 10) * 10 AS decade, COUNT(*) AS cnt
                    FROM users GROUP BY (age / 10) * 10
                )
                SELECT * FROM age_groups ORDER BY decade
            """)
            for row in cur.fetchall():
                print(dict(row))
            # JSONB filter
            cur.execute("SELECT * FROM users WHERE metadata @> %s", ('{"updated": true}',))
            for row in cur.fetchall():
                print(dict(row))
    finally:
        conn.close()


def pg_transaction_example():
    """Postgres: Transaction with commit/rollback."""
    if not PSYCOPG2_AVAILABLE:
        return
    conn = None
    try:
        conn = get_pg_conn()
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (name, age, email) VALUES (%s, %s, %s)", ("Tx User", 22, "tx@example.com"))
            conn.commit()
            cur.execute("UPDATE users SET age = 23 WHERE email = %s", ("tx@example.com",))
            # conn.rollback()  # uncomment to rollback
            conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print("Rolled back:", e)
    finally:
        if conn:
            conn.close()


# =============================================================================
# MILVUS VECTOR DB CRUD (basic to advanced)
# =============================================================================

VECTOR_DIM = 128  # dimension of embedding vectors

def milvus_ensure_collection(milvus_client=None):
    """Create Milvus collection if not exists (vector-only; for scalar fields define schema)."""
    if not PYMILVUS_AVAILABLE:
        print("pip install pymilvus")
        return
    client = milvus_client or get_milvus_client()
    if not client.has_collection(MILVUS_COLLECTION):
        client.create_collection(
            collection_name=MILVUS_COLLECTION,
            dimension=VECTOR_DIM,
            metric_type="IP",  # Inner Product (cosine-like when normalized); or "L2"
            auto_id=True,
            description="Practice collection for vector CRUD",
        )
        print(f"Created collection: {MILVUS_COLLECTION}")
    return client


def milvus_basic_insert():
    """Milvus: Insert vectors."""
    if not PYMILVUS_AVAILABLE:
        return
    import random
    client = milvus_ensure_collection()
    vectors = [[random.random() for _ in range(VECTOR_DIM)] for _ in range(5)]
    data = [{"vector": v} for v in vectors]
    res = client.insert(collection_name=MILVUS_COLLECTION, data=data)
    print("Inserted:", res)


def milvus_basic_search():
    """Milvus: Vector similarity search (k-NN)."""
    if not PYMILVUS_AVAILABLE:
        return
    import random
    client = get_milvus_client()
    query_vector = [[random.random() for _ in range(VECTOR_DIM)]]
    res = client.search(
        collection_name=MILVUS_COLLECTION,
        data=query_vector,
        limit=3,
    )
    for hits in res:
        for h in hits:
            print("id:", h["id"], "distance:", h["distance"])


def milvus_query():
    """Milvus: Query by expression (e.g. primary key)."""
    if not PYMILVUS_AVAILABLE:
        return
    client = get_milvus_client()
    # Query by id (use ids returned from insert)
    res = client.query(
        collection_name=MILVUS_COLLECTION,
        filter="id >= 0",
        limit=10,
        output_fields=["*"],
    )
    for r in res:
        print(r)


def milvus_delete():
    """Milvus: Delete by expression (e.g. id in [1,2,3])."""
    if not PYMILVUS_AVAILABLE:
        return
    client = get_milvus_client()
    client.delete(collection_name=MILVUS_COLLECTION, filter="id in [1, 2, 3]")
    print("Deleted entities matching filter")


def milvus_advanced_hybrid_search():
    """Milvus: Search with optional filter (vector + expr when schema has scalar fields)."""
    if not PYMILVUS_AVAILABLE:
        return
    import random
    client = get_milvus_client()
    query_vector = [[random.random() for _ in range(VECTOR_DIM)]]
    res = client.search(
        collection_name=MILVUS_COLLECTION,
        data=query_vector,
        limit=5,
    )
    for hits in res:
        for h in hits:
            print(h)


def milvus_get_stats():
    """Milvus: Get collection stats."""
    if not PYMILVUS_AVAILABLE:
        return
    client = get_milvus_client()
    stats = client.get_collection_stats(MILVUS_COLLECTION)
    print("Collection stats:", stats)


# =============================================================================
# RUN EXAMPLES (uncomment to execute)
# =============================================================================

if __name__ == "__main__":
    # ----- MONGO -----
    # basic_create()
    # basic_read()
    # basic_update()
    # basic_delete()
    # intermediate_read()
    # intermediate_update()
    # intermediate_delete()
    # advanced_read()
    # advanced_update()
    # advanced_delete()
    # bulk_write()
    # aggregation_example()
    # find_one_and_operations()

    # ----- POSTGRES (set POSTGRES_URI or POSTGRES_* env vars) -----
    # pg_ensure_table(get_pg_conn())
    # pg_basic_create()
    # pg_basic_read()
    # pg_basic_update()
    # pg_basic_delete()
    # pg_intermediate_read()
    # pg_intermediate_update()
    # pg_advanced_read()
    # pg_transaction_example()

    # ----- MILVUS (set MILVUS_URI, run Milvus server) -----
    # milvus_ensure_collection()
    # milvus_basic_insert()
    # milvus_basic_search()
    # milvus_query()
    # milvus_delete()
    # milvus_advanced_hybrid_search()
    # milvus_get_stats()

    # Quick Mongo test
    collection.insert_one({"name": "Test User", "age": 99, "email": "test@example.com"})
    for doc in collection.find():
        print(doc)