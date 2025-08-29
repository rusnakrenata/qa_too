WITH vars AS (
    SELECT 36 AS run_configs_id, 1 AS iteration_id
),
routes_with_min AS (
    SELECT 
        vr.run_configs_id,
        vr.iteration_id,
        vr.vehicle_id,
        MIN(vr.duration) AS min_duration
    FROM vehicle_routes vr
    JOIN vars ON vr.run_configs_id = vars.run_configs_id
             AND vr.iteration_id = vars.iteration_id
    GROUP BY vr.run_configs_id, vr.iteration_id, vr.vehicle_id
),
penalties AS (
    SELECT
        a.run_configs_id,
        a.iteration_id,
        a.vehicle_id,
        a.route_id,
        a.duration - b.min_duration AS penalty
    FROM vehicle_routes a
    JOIN routes_with_min b 
        ON a.vehicle_id = b.vehicle_id
       AND a.run_configs_id = b.run_configs_id
       AND a.iteration_id = b.iteration_id
),
qa AS (
    SELECT SUM(p.penalty) AS qa_penalty
    FROM qa_selected_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
),
gurobi AS (
    SELECT SUM(p.penalty) AS gurobi_penalty
    FROM gurobi_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
),
random AS (
    SELECT SUM(p.penalty) AS random_penalty
    FROM random_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
),
cong AS (
    SELECT 
        SUM(cs.congestion_post_qa) AS cong_post_qa,
        SUM(cs.congestion_post_gurobi) AS cong_post_gurobi,
        SUM(cs.congestion_random) AS cong_random,
        SUM(cs.congestion_shortest_dur) AS cong_shortest_dur
    FROM trafficOptimization.congestion_summary cs
    JOIN vars 
        ON cs.run_configs_id = vars.run_configs_id
       AND cs.iteration_id = vars.iteration_id
)
SELECT 
    qa.qa_penalty,
    gurobi.gurobi_penalty,
    random.random_penalty,
    0 AS shortest_penalty,
    cong.*,
    qa.qa_penalty + cong.cong_post_qa AS QA_COST,
    gurobi.gurobi_penalty + cong.cong_post_gurobi AS GUROBI_COST,
    random.random_penalty + cong.cong_random AS RANDOM_COST,
    0 + cong.cong_shortest_dur AS SHORTEST_DUR_COST,
    gr.objective_value,
    qr.energy,
    qr.assignment_valid
FROM qa
JOIN gurobi ON 1=1
JOIN random ON 1=1
JOIN cong ON 1=1
JOIN vars ON 1=1
JOIN gurobi_results gr ON gr.run_configs_id = vars.run_configs_id
					AND gr.iteration_id  = vars.iteration_id
JOIN qa_results qr ON qr.run_configs_id = vars.run_configs_id
					AND qr.iteration_id  = vars.iteration_id;
