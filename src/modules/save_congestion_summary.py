from typing import Any, Optional
import pandas as pd
from models import CongestionSummary
from sqlalchemy.orm import Session

def save_congestion_summary(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    edges_df: pd.DataFrame,
    congestion_df: pd.DataFrame,
    post_qa_congestion_df: pd.DataFrame,
    shortest_routes_dur_df: pd.DataFrame,
    shortest_routes_dis_df: pd.DataFrame,
    random_routes_df: pd.DataFrame,
    post_gurobi_df: pd.DataFrame,
    sa_df: Optional[pd.DataFrame] = None,
    tabu_df: Optional[pd.DataFrame] = None,
    cbc_df: Optional[pd.DataFrame] = None,
) -> None:
    """Save congestion summary to the database (always accepts SA/TABU/CBC, even if empty)."""

    def _prep(df: Optional[pd.DataFrame], out_col: str) -> pd.DataFrame:
        """Normalize a congestion df to (edge_id, <out_col>) and allow empty/None."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["edge_id", out_col])
        # If caller passed raw per-edge rows, group just in case
        if "edge_id" in df and "congestion_score" in df:
            g = df.groupby("edge_id", as_index=False).agg({"congestion_score": "sum"})
            g = g.rename(columns={"congestion_score": out_col})
            return g[["edge_id", out_col]]
        # If already renamed outside, keep it
        cols = [c for c in df.columns if c.lower() in {out_col.lower(), "congestion_score"}]
        if "edge_id" in df and cols:
            x = df[["edge_id", cols[0]]].copy()
            x = x.rename(columns={cols[0]: out_col})
            return x
        return pd.DataFrame(columns=["edge_id", out_col])

    # 1) Aggregate base congestion
    congestion_df_grouped = (
        congestion_df.groupby("edge_id", as_index=False)
        .agg({"congestion_score": "sum"})
        .rename(columns={"congestion_score": "congestion_all"})
    )

    # 2) Initialize merged with all edges
    pd.set_option('future.no_silent_downcasting', True)
    merged = pd.DataFrame({"edge_id": edges_df.drop_duplicates(subset="edge_id")["edge_id"]})

    # 3) Merge all congestion variants (safe even if empty)
    merged = merged.merge(congestion_df_grouped, on="edge_id", how="left")
    merged = merged.merge(_prep(post_qa_congestion_df, "congestion_post_qa"), on="edge_id", how="left")
    merged = merged.merge(_prep(shortest_routes_dur_df, "congestion_shortest_dur"), on="edge_id", how="left")
    merged = merged.merge(_prep(shortest_routes_dis_df, "congestion_shortest_dis"), on="edge_id", how="left")
    merged = merged.merge(_prep(random_routes_df, "congestion_random"), on="edge_id", how="left")
    merged = merged.merge(_prep(post_gurobi_df, "congestion_post_gurobi"), on="edge_id", how="left")
    merged = merged.merge(_prep(sa_df, "congestion_post_sa"), on="edge_id", how="left")
    merged = merged.merge(_prep(tabu_df, "congestion_post_tabu"), on="edge_id", how="left")
    merged = merged.merge(_prep(cbc_df, "congestion_post_cbc"), on="edge_id", how="left")

    # 4) Fill NaNs with 0
    merged = merged.fillna(0).infer_objects(copy=False)


    # 5) Convert to ORM records
    records = [
        CongestionSummary(
            run_configs_id=run_config_id,
            iteration_id=iteration_id,
            edge_id=int(row["edge_id"]),
            congestion_all=float(row["congestion_all"]),
            congestion_post_qa=float(row["congestion_post_qa"]),
            congestion_post_sa=float(row["congestion_post_sa"]),
            congestion_post_tabu=float(row["congestion_post_tabu"]),
            congestion_shortest_dur=float(row["congestion_shortest_dur"]),
            congestion_shortest_dis=float(row["congestion_shortest_dis"]),
            congestion_random=float(row["congestion_random"]),
            congestion_post_gurobi=float(row["congestion_post_gurobi"]),
            congestion_post_cbc=float(row["congestion_post_cbc"]),
        )
        for _, row in merged.iterrows()
    ]

    # 6) Store to DB
    try:
        session.add_all(records)
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to commit congestion summary to the database.") from e
    finally:
        session.close()
