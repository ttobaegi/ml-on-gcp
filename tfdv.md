## TFDV
```
[methods for methods in dir(tfdv)]
```
```
['CombinerStatsGenerator',
 'FeaturePath',
 'GenerateStatistics',
 'StatsOptions',
 'TransformStatsGenerator',
 'WriteStatisticsToBinaryFile',
 'WriteStatisticsToTFRecord',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 '__version__',
 'anomalies',
 'api',
 'arrow',
 'coders',
 'compare_slices',
 'constants',
 'display_anomalies',
 'display_schema',
 'experimental_get_feature_value_slicer',
 'generate_statistics_from_csv',
 'generate_statistics_from_dataframe',
 'generate_statistics_from_tfrecord',
 'get_domain',
 'get_feature',
 'get_feature_stats',
 'get_slice_stats',
 'infer_schema',
 'load_anomalies_text',
 'load_schema_text',
 'load_statistics',
 'load_stats_binary',
 'load_stats_text',
 'pywrap',
 'set_domain',
 'statistics',
 'types',
 'update_schema',
 'utils',
 'validate_examples_in_csv',
 'validate_examples_in_tfrecord',
 'validate_statistics',
 'version',
 'visualize_statistics',
 'write_anomalies_text',
 'write_schema_text',
 'write_stats_text']
 ```
 
### Compute and visualize statistics

First we'll use [`tfdv.generate_statistics_from_csv`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/generate_statistics_from_csv) to compute statistics for our training data. (ignore the snappy warnings)

TFDV can compute descriptive [statistics](https://github.com/tensorflow/metadata/blob/v0.6.0/tensorflow_metadata/proto/v0/statistics.proto) that provide a quick overview of the data in terms of the features that are present and the shapes of their value distributions.

Internally, TFDV uses [Apache Beam](https://beam.apache.org/)'s data-parallel processing framework to scale the computation of statistics over large datasets. For applications that wish to integrate deeper with TFDV (e.g., attach statistics generation at the end of a data-generation pipeline), the API also exposes a Beam PTransform for statistics generation.

**NOTE:  Compute statistics**
* [tfdv.generate_statistics_from_csv](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/generate_statistics_from_csv)
* [tfdv.generate_statistics_from_dataframe](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/generate_statistics_from_dataframe)
* [tfdv.generate_statistics_from_tfrecord](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/generate_statistics_from_tfrecord)
