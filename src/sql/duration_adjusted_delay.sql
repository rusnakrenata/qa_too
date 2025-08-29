WITH cong as (
SELECT run_configs_id,
iteration_id,
SUM(congestion_post_qa ) as cong_post_qa,
SUM(congestion_post_gurobi) as cong_post_gurobi,
SUM(congestion_random) as cong_random,
SUM(congestion_shortest_dur) as cong_shortest_dur
FROM trafficOptimization.congestion_summary cs 
WHERE run_configs_id = 24
  AND iteration_id = 1
)
SELECT 
	SUM(post_qa_dur) + SUM(cong_post_qa)*10*0.5 as post_qa_DUR_ADJ, 
	SUM(post_gurobi_dur) + SUM(cong_post_gurobi)*10*0.5 as post_gurobi_DUR_ADJ, 
	SUM(shortest_dur) + SUM(cong_shortest_dur)*10*0.5 as shortest_DUR_ADJ,
	SUM(rnd_dur) + SUM(cong_random)*10*0.5 as rnd_DUR_ADJ,
    SUM(shortest_dur) AS shortest_dur,
    SUM(post_qa_dur) AS post_qa_dur,
    SUM(post_gurobi_dur) as post_gurobi_dur,
    SUM(rnd_dur) AS rnd_dur,
    SUM(cong_shortest_dur) as cong_shortest_dur,
    SUM(cong_post_qa) as cong_post_qa,
    SUM(cong_post_gurobi) as cong_post_gurobi,
    SUM(cong_random) as cong_random
FROM trafficOptimization.dist_dur_summary dd
JOIN cong cs on  dd.run_configs_id = cs.run_configs_id
  and dd.iteration_id= cs.iteration_id 
  



  
