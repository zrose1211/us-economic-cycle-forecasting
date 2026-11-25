-- 1. Unemployment and output growth in each decade (descriptive, not causal)
SELECT CAST(year / 10 AS INTEGER) * 10 AS decade,
       COUNT(*) AS quarters,
       ROUND(AVG(unemp), 2) AS mean_unemployment_pct,
       ROUND(AVG(gdp_growth_qoq), 2) AS mean_qoq_gdp_growth_pct
FROM forecast_features
GROUP BY CAST(year / 10 AS INTEGER) * 10
ORDER BY decade;

-- 2. Sharpest recorded changes in the unemployment rate (relative to prior quarter)
SELECT period, ROUND(unemp_change_prev, 2) AS unemployment_change_pp,
       ROUND(gdp_growth_qoq, 2) AS gdp_growth_qoq_pct,
       ROUND(tbilrate, 2) AS treasury_bill_rate_pct
FROM forecast_features
ORDER BY ABS(unemp_change_prev) DESC
LIMIT 10;

-- 3. Historical rates and next-quarter changes by broad rate band
SELECT CASE WHEN tbilrate < 3 THEN 'Below 3%'
            WHEN tbilrate < 7 THEN '3%–7%'
            ELSE '7% or higher' END AS t_bill_rate_band,
       COUNT(*) AS quarters,
       ROUND(AVG(target_next_unemp - unemp), 3) AS avg_next_q_unemployment_change_pp
FROM forecast_features
GROUP BY t_bill_rate_band
ORDER BY t_bill_rate_band;
