WITH 
vehicle_routes AS (
    SELECT 
        vehicle_id, 
        route_id, 
        distance, 
        duration,
        run_configs_id,
        iteration_id
    FROM vehicle_routes vr 
    WHERE vr.run_configs_id = %s 
      AND vr.iteration_id = %s
),
posta_qa AS (
    SELECT 
        sr.vehicle_id, 
        sr.route_id, 
        vr.distance, 
        vr.duration
    FROM qa_selected_routes sr 
    INNER JOIN vehicle_routes vr 
        ON sr.vehicle_id = vr.vehicle_id
        AND sr.route_id = vr.route_id
    WHERE sr.run_configs_id = vr.run_configs_id
      AND sr.iteration_id = vr.iteration_id
),
posta_sa AS (
    SELECT 
        sr.vehicle_id, 
        sr.route_id, 
        vr.distance, 
        vr.duration
    FROM sa_selected_routes sr 
    INNER JOIN vehicle_routes vr 
        ON sr.vehicle_id = vr.vehicle_id
        AND sr.route_id = vr.route_id
    WHERE sr.run_configs_id = vr.run_configs_id
      AND sr.iteration_id = vr.iteration_id
),
posta_tabu AS (
    SELECT 
        sr.vehicle_id, 
        sr.route_id, 
        vr.distance, 
        vr.duration
    FROM tabu_selected_routes sr 
    INNER JOIN vehicle_routes vr 
        ON sr.vehicle_id = vr.vehicle_id
        AND sr.route_id = vr.route_id
    WHERE sr.run_configs_id = vr.run_configs_id
      AND sr.iteration_id = vr.iteration_id
),
post_gurobi AS (
    SELECT 
        sr.vehicle_id, 
        sr.route_id, 
        vr.distance, 
        vr.duration
    FROM gurobi_routes sr 
    INNER JOIN vehicle_routes vr 
        ON sr.vehicle_id = vr.vehicle_id
        AND sr.route_id = vr.route_id
    WHERE sr.run_configs_id = vr.run_configs_id
      AND sr.iteration_id = vr.iteration_id
),
random AS (
    SELECT 
        rr.vehicle_id, 
        rr.route_id, 
        vr.distance, 
        vr.duration
    FROM random_routes rr 
    INNER JOIN vehicle_routes vr 
        ON rr.vehicle_id = vr.vehicle_id
        AND rr.route_id = vr.route_id
    WHERE rr.run_configs_id = vr.run_configs_id
      AND rr.iteration_id = vr.iteration_id
),
shortest AS (
    SELECT 
        vehicle_id, 
        MIN(duration) AS min_dur, 
        MIN(distance) AS min_dist 
    FROM vehicle_routes vr 
    GROUP BY vehicle_id
)
SELECT 
    SUM(s.min_dist) AS shortest_dist, 
    SUM(s.min_dur) AS shortest_dur,
    SUM(p.distance) AS post_qa_dist, 
    SUM(p.duration) AS post_qa_dur,
    SUM(sa.distance) AS post_sa_dist, 
    SUM(sa.duration) AS post_sa_dur,    
    SUM(ta.distance) AS post_tabu_dist, 
    SUM(ta.duration) AS post_tabu_dur,
    SUM(r.distance) AS rnd_dist, 
    SUM(r.duration) AS rnd_dur,
    SUM(gr.distance) as post_gurobi_dist,
    SUM(gr.duration) as post_gurobi_dur
FROM shortest s
INNER JOIN posta_qa p 
    ON p.vehicle_id = s.vehicle_id
INNER JOIN post_gurobi gr
    ON gr.vehicle_id = s.vehicle_id
INNER JOIN random r 
    ON r.vehicle_id = s.vehicle_id
INNER JOIN posta_sa sa
    ON sa.vehicle_id = s.vehicle_id
INNER JOIN posta_tabu ta
    ON ta.vehicle_id = s.vehicle_id;
