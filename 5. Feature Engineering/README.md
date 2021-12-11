# _5.	Feature Engineering_


## _BQML_
> - Remove examples that you don’t want to train on.
> - Compute vocabularies for categorical columns.
> - Compute aggregate statistics for numeric columns.
> - Consider advanced feature engineering using
> - `ML.FEATURE_CROSS`, `TRANSFORM`, and `BUCKETIZE`

- `ML.FEATURE_CROSS(STRUCT(features))` does a feature cross of all the combinations.
- `ML.POLYNOMIAL_EXPAND(STRUCT(features), degree)` creates x, x2, x3, etc.
- ML.BUCKETIZE(f, split_points) where split_points is an array


## _Apache Beam/Cloud Dataflow_


## _tf.data_
Wrap the dataframe with `tf.data` ; use feature columns as bridge to map from the columns in the Pandas dataframe to features use



## _tf.transform_

- 3 Possible places to do feature engineering 

   ![image](https://user-images.githubusercontent.com/79742748/145672165-3d376ce1-2176-47ff-add6-aa5cc09355ed.png)
    
    - Running a Dataflow pipeline in prediction seems a bit like overkill.

- tf.transform is a part of TFX, component used to analyze and transform training data
  
  ![image](https://user-images.githubusercontent.com/79742748/145674838-746899fb-1382-448b-9b1c-ba7cc48ee6a4.png)


#### `tf.transform` 


- Implement feature preprocessing and feature creation / Carry out feature processing efficiently, at scale and on streaming data.
- Define preprocessing pipelines and run these using large scale data processing frameworks, while also exporting the pipeline in a way that can be run as part of a TensorFlow graph.

> -   _Preproccess data and engineer new features using `Tf.Transform`_
> -   _Create and deploy **Apache Beam** pipeline_
> -   _Use processed data to train taxifare model locally then serve a prediction_
