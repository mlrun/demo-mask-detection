import mlrun
from kfp import dsl


@dsl.pipeline(name="Mask Detection Pipeline")
def kfpipeline(
    archive_url: str,
    dataset_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    download_data_flag: int = 1
):
    # Get our project object:
    project = mlrun.get_current_project()

    ###########################################################
    ###############    Download the dataset:    ###############
    ###########################################################
    # Download only if needed (meaning if 'download_data_flag' = 1):
    with dsl.Condition(download_data_flag == 1) as download_data_condition:
        # Run it using the 'open_archive' handler:
        open_archive_run = mlrun.run_function(
            function="open-archive",
            handler="open_archive",
            name="download_data",
            inputs={"archive_url": archive_url},
            params={"target_path": dataset_path}
        )

    ####################################################
    ###############    Train a model:    ###############
    ####################################################    
    # Run it using our 'train' handler:
    training_run = mlrun.run_function(
        function="training-and-evaluation",
        handler="train",
        name="training",
        params={
            "dataset_path": dataset_path,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "model_name": "mask_detector_pipeline"
        },
        outputs=["model"]
    ).after(download_data_condition)

    ###################################################################
    ###############    Convert to ONNX and optimize:    ###############
    ###################################################################
    # Run it using the 'to_onnx' handler:
    to_onnx_run = mlrun.run_function(
        function="onnx-utils",
        handler="to_onnx",
        name="optimizing",
        params={
            "model_path": training_run.outputs['model'],
            "onnx_model_name": 'onnx_mask_detector'
        },
        outputs=["model"],
    )

    #########################################################
    ###############    Evaluate the model:    ###############
    #########################################################
    # Run it using our 'evaluate' handler:
    evaluation_run = mlrun.run_function(
        function="training-and-evaluation",
        handler="evaluate",
        name="evaluating",
        params={
            "model_path": training_run.outputs['model'],
            "dataset_path": dataset_path,
            "batch_size": batch_size
        }
    )

    ################################################################################
    ###############    Deploy the model as a serverless function:    ###############
    ################################################################################
    # Get the function:
    serving_function = project.get_function("serving")
    # Increase the time limit as the image may take long time to be downloaded:
    serving_function.spec.readiness_timeout = 60 * 20  # 20 minutes.
    # Set the topology and get the graph object:
    graph = serving_function.set_topology("flow", engine="async")
    # Build the serving graph:
    graph.to(handler="resize", name="resize")\
         .to(handler="preprocess", name="preprocess")\
         .to(class_name="mlrun.frameworks.onnx.ONNXModelServer", name="onnx_mask_detector", model_path=str(to_onnx_run.outputs["model"]))\
         .to(handler="postprocess", name="postprocess").respond()
    # Set the desired requirements:
    serving_function.with_requirements(requirements=["onnxruntime~=1.14.0", "onnxoptimizer~=0.3.0", "protobuf<=3.20.1"])
    # Deploy the serving function:
    mlrun.deploy_function("serving")