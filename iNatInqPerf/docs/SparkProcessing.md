# Spark For iNaturalist Data

We want to take the CSV data in the S3 buckets on iNaturalist's accounts and perform various operations to upload them to a vector database.

The iNaturalist data has various CSV files, all of which are required for the final data processing, but that is a problem to be tackled in the future.

## tl;dr

The key takeaway from this investigation is that Apache Spark with Hadoop provides us with all the blocks we need to build a full parallelized ingestion engine without the need for any special or third-party extensions/plugins.

## CSVs and Structure

There are a total of 6 CSV files:

- observations.csv
- observers.csv
- photos.csv
- taxa.csv
- observations_projects.csv
- projects.csv

The one that is currently revelant to us is `photos.csv`. It has the following data structure with types corresponding to Postgres data types:

```yaml
photos:
  photo_uuid: uuid
  photo_id: integer
  observation_uuid: uuid
  observer_id: integer
  extension: character
  license: character
  width: smallint
  height: smallint
  position: smallint
```

## S3 Metadata

The recommended approach for handling S3 metadata would be to use the [boto3 package](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) which is AWS's official SDK for interacting with S3.

The documentation provides a [good list of examples](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-examples.html) which can be adapted to our needs.

## Apache Spark

We are using `pyspark` which is the Python package for interacting with the Apache Spark library.

### Reading CSVs in S3

The basic building blocks for using Spark to read CSVs from S3 are the following:

1. `pyspark.sql` which helps creats a session which we can use to connect to the S3 account. Normally this would require an AWS authkey, but since the iNaturalist data is open, it should be unnecessary.
2. Apache Hadoop: Hadoop is a data storage and processing framework which works with Spark. Hadoop provides us with the abstraction layer to connect to S3 buckets and read the CSV.
3. We can use `boto3` to aid in reading the metadata within the S3 bucket, such as listing all the files, etc. This however doesn't seem to be necessary since the data is well defined, but this is a good-to-know.

Note that Spark will create a Hadoop-based Resilient Distributed Dataset (RDD) which will allow for parallelizing operations in the next phase.

[Spark Docs for External Datasets](https://spark.apache.org/docs/latest/rdd-programming-guide.html#external-datasets)

[Spark CSV Docs](https://spark.apache.org/docs/latest/sql-data-sources-csv.html)

A helpful blog post with details can be found [here](https://medium.com/@satrupapanda1/step-by-step-guide-for-reading-data-from-s3-using-pyspark-140a99fb19ba).

### Processing the Data

Once we have read the CSV data into a DataFrame (Spark's primary data structure), we can process the data.
The major steps involves would be:

1. Initialize the Spark context which allows for multiprocessing.
2. Define a `foreach` transformation operation which will read a row of the CSV, get the photo ID and retrieve the corresponding vector embedding.
3. Use the `VectorDatabase` driver to upload the data point to the vector database.

[Spark Actions Docs](https://spark.apache.org/docs/latest/rdd-programming-guide.html#actions)

### Fault Tolerance

Due to the massive scale of the data (400M+ images), fault tolerance when running these operations is of critical importance.

Fortunately, Spark's RDD data abstraction provides built-in fault tolerance for various aspects such as Driver/Worker Node failure and referencing a dataset from external storage (such as an S3 bucket).

Hevo Data has a [very nice article](https://hevodata.com/learn/spark-fault-tolerance/) providing a good overview with sufficient details.

Spark will monitor failures and try to re-run jobs which have failed. The `spark.task.maxFailures` configuration lets us control the number of retries.

The part we will have to build out is a robust logging system which will record failures and help with debugging.
This needs to be done with particular attention since error logging has been a challenge for us until this point.

#### Checkpointing

Spark provides checkpointing capabilities to allow for recovery from failed tasks when used in the __streaming__ context.
Since we would have the S3 data pre-loaded via a CSV read, this may not be useful, however the use of Spark Streaming is an open design decision if we decide that is the way to go.

To enable effective checkpointing, we need to ensure the following:

1. Metadata checkpointing: This helps recover from processing failures such as ETL transformations.
2. Data checkpointing: This for saving generated RDDs (as a result of some __stateful__ transformation) to a persistent storage. From the current design outlined in this document, we are not performing stateful transformations hence this might not be required but is a good-to-have.

The docs to configure checkpointing can be found [here](https://spark.apache.org/docs/latest/streaming-programming-guide.html#how-to-configure-checkpointing).
This looks like a simple call to `streamingContext.checkpoint(checkpointDirectory)`, where `streamingContext` is the context created on job startup.

## Hardware Specifications

Since Spark uses a driver-work parallelization model, this is highly dependent on the number of CPUs and threads available to Spark.

For a quick introduction along with a discussion of various considerations, [this blog post](https://medium.com/@harshita.motwani23/parallelism-in-azure-databricks-process-multiple-data-at-scale-153aa3c03442) is a good resource.


Azure supports job parallelization in a cluster through the use of [Azure Databricks](https://azure.microsoft.com/en-us/products/databricks).
A good practice might be to leverage Databricks as Spark's underlying RDD implementation as a [Data Lakehouse](https://learn.microsoft.com/en-us/azure/databricks/introduction/#build-an-enterprise-data-lakehouse).

This requires a cost/performance trade-off which depends on the size of the cluster and the power of the provisioned CPUs within the cluster. At a high level, horizontal scalability with a large number of mid-performance VMs might be the most effective middle ground.

## Azure Databricks Lakeflow Connect

We can use [Databricks Lakeflow Connect](https://learn.microsoft.com/en-us/azure/databricks/ingestion/) as the common service built upon Spark to manage both the loading the S3/DBMS data to the data lakehouse and the ETL required to transform the data and push it to the vector database.

Databricks provides an [excellent reference documentation page](https://learn.microsoft.com/en-us/azure/databricks/ingestion/) for performing data ingestion and transformation.

The [AutoLoader](https://learn.microsoft.com/en-us/azure/databricks/ingestion/cloud-object-storage/auto-loader/patterns#pipeline-syntax) provides the means to perform Spark-based connectivity to S3.