# Distribution of vehicles with different number of intersepced vehicles
Select countvehicle,count(distinct vehicle1)
FROM (
SELECT  vehicle1,count(distinct vehicle2) as countvehicle
FROM trafficOptimization.congestion_map cm 
where iteration_id = 3 and run_configs_id = 7
#and congestion_score  <= 0
group by vehicle1) as tab1
group by countvehicle
order by countvehicle asc