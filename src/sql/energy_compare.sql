select distinct
	round(qrs.n_filtered_vehicles/50)* 50 as rounded_vehicles,
	count(*),
	avg(qr.energy) as qa_energy,
	avg(gr.objective_value) as gurobi_energy,
	avg(sr.energy) as sa_energy,
	avg(tr.energy) as tabu_energy,
	avg(case when cr.status = 'Optimal' then cr.objective_value else NULL end) as cbc_energy
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
left join sa_results sr on qr.run_configs_id = sr.run_configs_id
				and qr.iteration_id = sr.iteration_id
				and qr.cluster_id = sr.cluster_id
left join tabu_results tr on qr.run_configs_id = tr.run_configs_id
				and qr.iteration_id = tr.iteration_id
				and qr.cluster_id = tr.cluster_id
inner join cbc_results cr on qr.run_configs_id = cr.run_configs_id
				and qr.iteration_id = cr.iteration_id
				and qr.cluster_id = cr.cluster_id
group by round(qrs.n_filtered_vehicles/50)* 50
order by 1