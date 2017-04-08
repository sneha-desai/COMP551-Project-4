import sqlite3
import pandas as pd

cnx = sqlite3.connect("compas.db")
df_charge = pd.read_sql_query("SELECT * FROM charge", cnx)
df_casearrest = pd.read_sql_query("SELECT * FROM casearrest", cnx)
df_compas = pd.read_sql_query("SELECT * FROM compas", cnx)
df_jailhistory = pd.read_sql_query("SELECT * FROM jailhistory", cnx)
df_people = pd.read_sql_query("SELECT * FROM people", cnx)
df_prisonhistory = pd.read_sql_query("SELECT * FROM prisonhistory", cnx)
df_summary = pd.read_sql_query("SELECT * FROM summary", cnx)

#convert DataFrames to CSV
df_charge.to_csv("output/charge.csv")
df_casearrest.to_csv("output/casearrest.csv")
df_compas.to_csv("output/compas.csv")
df_jailhistory.to_csv("output/jailhistory.csv")
df_people.to_csv("output/people.csv")
df_prisonhistory.to_csv("output/prisonhistory.csv")
df_summary.to_csv("output/summary.csv")
