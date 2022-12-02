# FROM pitching

result = pd.read_sql_query(query1, db)

result

#%%
#Exercise 3
## Using a SQL query, select all rows in the same table where HR is less than 10 and gs is greater than 25. 
query2 = '''SELECT *
            FROM pitching
            WHERE HR < 10
                AND gs > 25

result = pd.read_sql_query(query2, db)

result'''

## Find out what the columns mean and explain your query in words
# Select the pitchers that allowed less than 10 homeruns and started more than 25 games in a team for that season. 

# %%
#Exercise 4
## Identify the shared columns (keys) and join the table in exercise 2 with the salaries table.
query3 = '''SELECT *
            FROM pitching
            JOIN salaries 
                USING (playerid, yearid, teamid)
            WHERE yearid = 1986'''

result = pd.read_sql_query(query3, db)

result
# We need to join using teamid because some pitchers transfered to another team 
# mid-season, and they are getting paid differently.

#%%
# Exercise 5

"""Though it’s not required by SQL, it is advisable to include all non-aggregated columns from your SELECT clause in your GROUP BY clause. If you don’t, there are cases where the query will return the desired results, there are also instances where a random value from the non-aggregated row will be used as the representative for all the values returned by the query.

For example, let’s say you wanted to know the average deal by sales agent for each of their customers. If you used the query:

SELECT sales_agent,
       account,
       SUM(close_value)
  FROM sales_pipeline
 WHERE sales_pipeline.deal_stage = "Won"
 GROUP BY sales_agent, account
 ORDER BY sales_agent, account

If you wanted to refine your query even more by running your aggregations against a limited set of the values in a column you could use the FILTER keyword. For example, if you wanted to know both the number of deals won by a sales agent and the number of those deals that had a value greater than 1000, you could use the query:

SELECT sales_agent,
       COUNT(sales_pipeline.close_value) AS total,
       COUNT(sales_pipeline.close_value)
FILTER(WHERE sales_pipeline.close_value > 1000) AS `over 1000`
  FROM sales_pipeline
 WHERE sales_pipeline.deal_stage = "Won"
 GROUP BY sales_pipeline.sales_agent


 As we saw in the aggregate functions section, WHERE also limits the values in a query against which an aggregate function is run. FILTER is more flexible than WHERE because you can use more than one FILTER modifier in an aggregate query while you can only use only one WHERE clause.

For example:

SELECT sales_agent,
       COUNT(sales_pipeline.close_value) AS `number won`,
       COUNT(sales_pipeline.close_value)
FILTER(WHERE sales_pipeline.close_value > 1000) AS `number won > 1000`,
       AVG(sales_pipeline.close_value) AS `average of all`,
       AVG(sales_pipeline.close_value)
FILTER(WHERE sales_pipeline.close_value > 1000) AS `avg > 1000`
  FROM sales_pipeline
 WHERE sales_pipeline.deal_stage = "Won"
 GROUP BY sales_pipeline.sales_agent
 
"""

SELECT sales_agent,
       AVG(close_value)
  FROM sales_pipeline
 WHERE sales_pipeline.deal_stage = "Won"
 GROUP BY sales_agent
 ORDER BY AVG(close_value) DESC

 SELECT sales_teams.manager,
       AVG(sales_pipeline.close_value)
  FROM sales_teams
       JOIN sales_pipeline ON (sales_teams.sales_agent = sales_pipeline.sales_agent)
 WHERE sales_pipeline.deal_stage = "Won"
 GROUP BY sales_teams.manager

## Create a query that captures the number of pitchers the Washington Nationals used in each year.

query4 = '''SELECT COUNT(DISTINCT playerid), yearid
            FROM pitching
            WHERE teamid = 'WAS'
            GROUP BY yearid
            ORDER BY yearid'''

result = pd.read_sql_query(query4, db)

result

#%%
# Exercise 6

## Research the order of operations for SQL and put the following keywords in that order.

# The order is FROM JOIN WHERE GROUP BY HAVING SELECT ORDER BY LIMIT
