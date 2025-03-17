# Training Detector2 models for segmenting breeding plots from UAV orthomosaics

This project involves developing a custom Detectron2 model to detect breeding plots in orthomosaics. The workflow incorporates Label Studio for image annotation, which integrates with a PostgreSQL for data management.



writable container

```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
apptainer shell \
   --writable \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   labelstudio_sandbox/
```

writable container

```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
apptainer shell \
   --writable \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   apptainer_sandbox/
```

writable container


```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
export TMPDIR=/mnt/cache
apptainer shell \
   --nv \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   apptainer_sandbox/
```

git clone https://github.com/HumanSignal/label-studio.git

# install dependencies
cd label-studio
pip install poetry
poetry install

# run db migrations
poetry run python label_studio/manage.py migrate

# collect static files
poetry run python label_studio/manage.py collectstatic

# start the server in development mode at http://localhost:8080
poetry run python label_studio/manage.py runserver

export DJANGO_DB=postgresql
export POSTGRE_NAME='UFPS'
export POSTGRE_USER='prabahar'
export POSTGRE_PASSWORD='Cera@3003'
export POSTGRE_HOST=localhost
export POSTGRE_PORT=5432



./configure --prefix=/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql
make
make install

/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql/bin/initdb -D /gpfs/fs7/aafc/phenocart/IDEs/Postgres/data
/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql/bin/pg_ctl -D /gpfs/fs7/aafc/phenocart/IDEs/Postgres/data -l logfile start -o "-p 5433"
/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql/bin/pg_ctl -D /gpfs/fs7/aafc/phenocart/IDEs/Postgres/data reload


kill $(pgrep postgres)

stop
/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql/bin/pg_ctl -D /gpfs/fs7/aafc/phenocart/IDEs/Postgres/data stop
# SemanticSegmentationwithDetectron2LabelStudioPostGRESDB
