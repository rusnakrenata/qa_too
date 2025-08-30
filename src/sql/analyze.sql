select qrs.n_vehicles,
	qrs.lambda_penalty,
	qrs.n_nodes_distinct,
	qrs.overall_overlap,
	qr.n_nodes_distinct,
	qr.overall_overlap,
	qr.energy,
	qr.n_vehicles_selected,
	qr.selected_vehicle_ids,
	gr.n_nodes_distinct,
	gr.overall_overlap,
	gr.objective_value,
	gr.`assignment`,
	gr.n_vehicles_selected,
	gr.selected_vehicle_ids
from qubo_run_stats qrs
inner join qa_results qr on  qrs.run_configs_id = qr.run_configs_id
		and qrs.iteration_id = qr.iteration_id
		and qrs.lambda_penalty = qr.lambda_value
inner join gurobi_results gr on qrs.run_configs_id = gr.run_configs_id
		and qrs.iteration_id = gr.iteration_id
		and qrs.lambda_penalty = gr.lambda_value
where qrs.run_configs_id = 70 and qrs.iteration_id = 8