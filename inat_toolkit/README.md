# inat_toolkit

Utilities to help with the iNaturalist Open Data dataset.

## Tips for setting up the dataset in PostgreSQL

- Make sure you have sufficient disk space. When you unzip the files, they balloon massively. At least 80 GB of free disk space is recommended.
- Increase the `maintenance_work_mem` memory. You can do so with the below command:

```sql
ALTER SYSTEM SET maintenance_work_mem = '512MB';
```

- Vacuum the tables to optimize space. This will make all the operations much faster.

```sql
VACUUM (ANALYZE, VERBOSE, FULL) photos;
VACUUM (ANALYZE, VERBOSE, FULL) observations;
VACUUM (ANALYZE, VERBOSE, FULL) observers;
VACUUM (ANALYZE, VERBOSE, FULL) taxa;
```

- Generate indexes for faster query performance.

```sql
CREATE INDEX index_photos_photo_id ON photos USING btree (photo_id);
CREATE INDEX index_photos_photo_uuid ON photos USING btree (photo_uuid);
CREATE INDEX index_photos_observation_uuid ON photos USING btree (observation_uuid);
CREATE INDEX index_taxa_taxon_id ON taxa USING btree (taxon_id);
CREATE INDEX index_observers_observer_id ON observers USING btree (observer_id);
CREATE INDEX index_observations_observer_id ON observations USING btree (observer_id);
CREATE INDEX index_observations_observation_uuid ON observations USING btree (observation_uuid);
CREATE INDEX index_observations_taxon_id ON taxa USING btree (taxon_id);
```