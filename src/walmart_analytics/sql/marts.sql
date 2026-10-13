-- name: revenue_by_year
SELECT year(date) AS year, COUNT(DISTINCT date) AS days,
       SUM(revenue) AS revenue, SUM(units) AS units
FROM sales GROUP BY 1 ORDER BY 1;

-- name: revenue_by_month_state
SELECT date_trunc('month', s.date) AS month, st.state_id, SUM(s.revenue) AS revenue
FROM sales s JOIN stores st USING (store_id)
GROUP BY 1, 2 ORDER BY 1, 2;

-- name: category_mix
SELECT i.cat_id, SUM(s.revenue) AS revenue, SUM(s.units) AS units,
       SUM(s.revenue) / SUM(SUM(s.revenue)) OVER () AS share
FROM sales s JOIN items i USING (item_id)
GROUP BY 1 ORDER BY 2 DESC;

-- name: store_scorecard
-- Full-year 2015 revenue per store, growth on 2014, and the share of
-- item-days that sold nothing.
WITH y AS (
    SELECT store_id, year(date) AS year, SUM(revenue) AS revenue,
           AVG((units = 0)::INT) AS zero_days
    FROM sales WHERE year(date) IN (2014, 2015) GROUP BY 1, 2
)
SELECT c.store_id, st.state_id, c.revenue AS revenue_2015,
       c.revenue / p.revenue - 1 AS growth_vs_2014, c.zero_days
FROM y c JOIN y p ON p.store_id = c.store_id AND p.year = 2014
JOIN stores st ON st.store_id = c.store_id
WHERE c.year = 2015 ORDER BY revenue_2015 DESC;

-- name: revenue_concentration
-- How many item-store combinations make 80% of revenue.
WITH r AS (SELECT store_id, item_id, SUM(revenue) AS rev FROM sales GROUP BY 1, 2),
c AS (
    SELECT rev, SUM(rev) OVER (ORDER BY rev DESC) AS cum, SUM(rev) OVER () AS total,
           ROW_NUMBER() OVER (ORDER BY rev DESC) AS rn, COUNT(*) OVER () AS n
    FROM r
)
SELECT MIN(rn) AS item_stores_for_80pct, MAX(n) AS item_stores,
       MIN(rn) * 1.0 / MAX(n) AS share
FROM c WHERE cum >= 0.8 * total;

-- name: weekday_index
SELECT c.weekday, AVG(d.revenue) / AVG(AVG(d.revenue)) OVER () AS index
FROM (SELECT date, SUM(revenue) AS revenue FROM sales GROUP BY 1) d
JOIN calendar c USING (date)
GROUP BY 1 ORDER BY index DESC;

-- name: snap_effect
-- FOODS revenue on SNAP days against non-SNAP days in the same state and month.
WITH d AS (
    SELECT st.state_id, s.date, SUM(s.revenue) AS revenue
    FROM sales s JOIN items i USING (item_id) JOIN stores st USING (store_id)
    WHERE i.cat_id = 'FOODS' GROUP BY 1, 2
),
f AS (
    SELECT d.*, date_trunc('month', d.date) AS month,
           CASE d.state_id WHEN 'CA' THEN c.snap_CA WHEN 'TX' THEN c.snap_TX ELSE c.snap_WI END AS snap
    FROM d JOIN calendar c USING (date)
),
m AS (
    SELECT state_id, month,
           AVG(revenue) FILTER (WHERE snap = 1) AS snap_rev,
           AVG(revenue) FILTER (WHERE snap = 0) AS other_rev
    FROM f GROUP BY 1, 2
)
SELECT state_id, AVG(snap_rev / other_rev - 1) AS snap_uplift, COUNT(*) AS months
FROM m WHERE snap_rev IS NOT NULL AND other_rev IS NOT NULL
GROUP BY 1 ORDER BY 1;

-- name: department_mix
SELECT i.dept_id, SUM(s.revenue) AS revenue,
       SUM(s.revenue) / SUM(SUM(s.revenue)) OVER () AS share
FROM sales s JOIN items i USING (item_id)
GROUP BY 1 ORDER BY 2 DESC;
