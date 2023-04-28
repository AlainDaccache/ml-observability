```
docker-compose up -d
gunicorn --statsd-host=localhost:8125 --statsd-prefix=helloworld --bind 127.0.0.1:8080 hello:app
```

```
python3 -m virtualenv --python=python3.8 venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Prometheus: localhost:9090/targets

Resources:
* https://dev.to/kirklewis/metrics-with-prometheus-statsd-exporter-and-grafana-5145
* https://medium.com/@damianmyerscough/monitoring-gunicorn-with-prometheus-789954150069

for i in `seq 100`; do curl http://127.0.0.1:5000/predict && sleep 1; done;

```python
import requests
import pandas as pd
import numpy as np
import random
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

os.environ["POSTGRES_USER"] = "myuser"
os.environ["POSTGRES_PASSWORD"] = "mypassword"
os.environ["POSTGRES_DB"] = "mydb"

# Extract the raw data and preprare it
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

df = pd.DataFrame(data, 
                  columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                              'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
df['MEDV'] = target

# Define the PostgreSQL connection URL
db_url = f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@localhost:5432/{os.environ['POSTGRES_DB']}"

# Create the engine
engine = create_engine(db_url)

# Insert the data into the table
df.to_sql('BOSTON_DATASET', engine, if_exists='append', index=False)

# Verify that the data has been inserted
with engine.connect() as conn:
    result = conn.execute('SELECT COUNT(*) FROM boston')
    print(f"Number of rows in boston table: {result.fetchone()[0]}")

requests.get("http://localhost:5000/train")


_, X_test, _, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

while True:
    rand = np.random.choice(X_test.shape[0], size=1, replace=False)
    sampled_row = data[rand]
    payload = {k: v for k, v in zip(raw_df.columns[:-1], sampled_row)}
    requests.get("http://localhost:5000/predict", json=payload)
```