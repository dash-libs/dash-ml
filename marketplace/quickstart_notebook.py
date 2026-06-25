# Databricks notebook source
# MAGIC %md
# MAGIC # dash-ml — ML Monitoring
# MAGIC
# MAGIC Monitor ML model drift and performance in production.
# MAGIC
# MAGIC **Install and launch:**

# COMMAND ----------

# MAGIC %pip install dash-ml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import dashml
dashml.launch()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Python API (optional — for automation)
# MAGIC
# MAGIC ```python
# MAGIC import dashml
# MAGIC # See docs/api/ for full API reference
# MAGIC ```
