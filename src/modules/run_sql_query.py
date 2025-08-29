import os
import pandas as pd

def run_sql_query(session, run_configs_id, iteration_id, sql_name='congestion_results.sql'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql_file = os.path.join(base_dir, 'sql', sql_name)
    with open(sql_file, 'r') as f:
        sql = f.read()
    # Split queries (naive split on ';')
    queries = [q.strip() for q in sql.split(';') if q.strip()]
    params = (run_configs_id, iteration_id)
    for i, query in enumerate(queries):
        print(f'--- Query {i+1} ---')
        try:
            df = pd.read_sql(query, session.bind, params=params)
            print(df)
        except Exception as e:
            print(f'Error running query {i+1}: {e}')