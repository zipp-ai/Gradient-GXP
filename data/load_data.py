import pandas as pd
from sqlalchemy import create_engine

# Set connection strings
LIMS_CONN = "postgresql://rhishibansode@localhost:5432/lims_db"
MES_CONN = "postgresql://rhishibansode@localhost:5432/mes_db"
QMS_CONN = "postgresql://rhishibansode@localhost:5432/qms_db"
CHANGE_CONTROL_CONN = "postgresql://rhishibansode@localhost:5432/change_control_db"

# Read CSVs
lims_df = pd.read_csv("data/comprehensive_lims_data.csv")
mes_df = pd.read_csv("data/comprehensive_mes_data.csv")
qms_df = pd.read_csv("data/comprehensive_qms_data.csv")
change_control_df = pd.read_csv("data/comprehensive_change_control_data.csv")


# Create engines
lims_engine = create_engine(LIMS_CONN)
mes_engine = create_engine(MES_CONN)
qms_engine = create_engine(QMS_CONN)
change_control_engine = create_engine(CHANGE_CONTROL_CONN)
# Load into PostgreSQL

lims_df.to_sql("lims_data", lims_engine, if_exists="replace", index=False)
mes_df.to_sql("mes_data", mes_engine, if_exists="replace", index=False)
qms_df.to_sql("qms_data", qms_engine, if_exists="replace", index=False)
change_control_df.to_sql("change_control_data", change_control_engine, if_exists="replace", index=False)

print("✅ Data loaded into lims_data, mes_data, qms_data, change_control_data tables.")
