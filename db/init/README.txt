Put your dataset dump here, renamed to 01-fnph-dataset.sql

Postgres runs every .sql file in this folder automatically, in
filename order, the FIRST time the "db" container starts with an
empty volume. If you've already run docker compose up once before
adding the file, run:

    docker compose down -v
    docker compose up --build

The -v flag wipes the postgres volume so the init script runs again.
