with base as (select 	
	round(qrs.n_filtered_vehicles/50)* 50 as rounded_vehicles,
	lambda_value ,
	max_weight,
	qr.energy,
	#qr.duration as qa_model_duration,
    #qr.solver_time as qa_solver_time,
	gr.objective_value,
	#gr.duration as gurobi_model_duration,
    #gr.solver_time as gr_solver_time,
    #gr.time_limit_seconds,
	#gr.best_bound,
	#gr.gap,
	FORMAT(qr.qubo_density * 100, 2) AS matrix_density,
    CONCAT(FORMAT((qr.energy - gr.objective_value)/NULLIF(ABS(gr.objective_value), 0) * 100,2), "%") as delta_energy
from qa_results qr 
inner join  gurobi_results gr on qr.run_configs_id = gr.run_configs_id
				and qr.iteration_id = gr.iteration_id
				and qr.cluster_id = gr.cluster_id
inner join qubo_run_stats qrs on qrs.run_configs_id = qr.run_configs_id
				and qrs.iteration_id = qr.iteration_id
				and qrs.cluster_id = qr.cluster_id
inner join iterations i on i.iteration_id = qr.iteration_id
				and i.run_configs_id = qr.run_configs_id
inner join run_configs rc on rc.run_configs_id = i.run_configs_id
inner join cities c on c.city_id =rc.city_id
where  comp_type = 'hybrid' )
select 
	*
from base
order by rounded_vehicles
