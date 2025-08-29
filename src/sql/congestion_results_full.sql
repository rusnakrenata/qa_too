SELECT 
    qr.n_vehicles, 
    qr.lambda_value, 
    qr.assignment_valid, 
    qr.invalid_assignment_vehicles,
    qr.qubo_size,
    qr.qubo_density,
    qr.energy as qa_energy,
    sa.energy as sa_energy,
    tr.energy as tabu_energy,
    cbc.objective_value as cbc_energy,
    cbc.status as cbc_status,
    gr.objective_value as gurobi_energy
FROM trafficOptimization.qa_results qr
INNER JOIN trafficOptimization.gurobi_results gr
    ON gr.run_configs_id = qr.run_configs_id
   AND gr.iteration_id = qr.iteration_id    
   AND gr.cluster_id = qr.cluster_id
INNER JOIN trafficOptimization.sa_results sa
    ON sa.run_configs_id = qr.run_configs_id
    AND sa.iteration_id = qr.iteration_id
    AND sa.cluster_id = qr.cluster_id
INNER JOIN trafficOptimization.tabu_results tr
    ON tr.run_configs_id = qr.run_configs_id
   AND tr.iteration_id = qr.iteration_id
   AND tr.cluster_id = qr.cluster_id
INNER JOIN trafficOptimization.cbc_results cbc
    ON cbc.run_configs_id = qr.run_configs_id
   AND cbc.iteration_id = qr.iteration_id
   AND cbc.cluster_id = qr.cluster_id
WHERE gr.run_configs_id = %s 
  AND gr.iteration_id = %s;
WITH vars AS (
    SELECT %s AS run_configs_id, %s AS iteration_id
), routes_with_min AS (
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
sa AS (
    SELECT SUM(p.penalty) AS sa_penalty
    FROM sa_selected_routes sr
    JOIN penalties p 
        ON p.vehicle_id = sr.vehicle_id
       AND p.route_id = sr.route_id
       AND p.run_configs_id = sr.run_configs_id
       AND p.iteration_id = sr.iteration_id
),
tabu AS (
    SELECT SUM(p.penalty) AS tabu_penalty
    FROM tabu_selected_routes sr
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
cbc AS (
    SELECT SUM(p.penalty) AS cbc_penalty
    FROM cbc_routes sr
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
        SUM(cs.congestion_post_sa) AS cong_post_sa,
        SUM(cs.congestion_post_tabu) AS cong_post_tabu,
        SUM(cs.congestion_post_gurobi) AS cong_post_gurobi,
        SUM(cs.congestion_post_cbc) AS cong_post_cbc,
        SUM(cs.congestion_random) AS cong_random,
        SUM(cs.congestion_shortest_dur) AS cong_shortest_dur
    FROM trafficOptimization.congestion_summary cs
    JOIN vars 
        ON cs.run_configs_id = vars.run_configs_id
       AND cs.iteration_id = vars.iteration_id
)
SELECT 
    qa.qa_penalty + cong.cong_post_qa AS QA_COST,
    sa.sa_penalty + cong.cong_post_sa AS SA_COST,
    tabu.tabu_penalty + cong.cong_post_tabu AS TABU_COST,
    gurobi.gurobi_penalty + cong.cong_post_gurobi AS GUROBI_COST,
    cbc.cbc_penalty + cong.cong_post_cbc AS CBC_COST,
    random.random_penalty + cong.cong_random AS RANDOM_COST,
    0 + cong.cong_shortest_dur AS SHORTEST_DUR_COST,
	cong.*
FROM qa
JOIN sa ON 1=1
JOIN tabu ON 1=1
JOIN gurobi ON 1=1
JOIN cbc ON 1=1
JOIN random ON 1=1
JOIN cong ON 1=1
JOIN vars ON 1=1;
WITH 
selected_routes AS (
    SELECT 
        vr.vehicle_id, 
        ss.route_id AS qa_route_id,
        gr.route_id as gurobi_route_id
    FROM vehicle_routes vr
    INNER JOIN qa_selected_routes ss 
        ON ss.vehicle_id = vr.vehicle_id
    INNER JOIN gurobi_routes gr 
        ON gr.vehicle_id = vr.vehicle_id
    WHERE vr.run_configs_id = ss.run_configs_id 
      AND vr.iteration_id = ss.iteration_id
      AND vr.run_configs_id = gr.run_configs_id 
      AND vr.iteration_id = gr.iteration_id
      AND vr.run_configs_id = %s
      AND vr.iteration_id = %s
    GROUP BY vr.vehicle_id
)
SELECT 
   vehicle_id, qa_route_id, gurobi_route_id
FROM selected_routes
WHERE gurobi_route_id <> qa_route_id;
 