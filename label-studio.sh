#!/usr/bin/env bash

# Start the PostgreSQL server
/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql/bin/pg_ctl -D /gpfs/fs7/aafc/phenocart/IDEs/Postgres/data -l logfilePostGRES start -o "-p 5432"

# If a screen session named "label-studio" exists, kill it
if screen -ls | grep -q "label-studio"; then
  screen -S label-studio -X quit
fi

# Activate the virtual environment
. /gpfs/fs7/aafc/phenocart/PhenomicsProjects/Detectron2/venv/bin/activate

# Start label-studio in a detached screen session named "label-studio"
screen -dmS label-studio label-studio
