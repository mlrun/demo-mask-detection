# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import mlrun
import os

def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    """
    Creating the project for this demo. This function is expected to be called automatically when
    calling the function `mlrun.get_or_create_project`.

    :returns: a fully prepared project for this demo.
    """
    # Set the project git source:
    source = project.get_param(key="source")
    default_image = project.get_param(key="default_image", default=None)
    build_image = project.get_param(key="build_image", default=False)
    use_gpu = project.get_param(key="use_gpu", default=False)
    framework = project.get_param(key="framework", default="tf-keras")

    if not source:
        source = "git://github.com/mlrun/demo-mask-detection.git"
    print(f"Project Source: {source}")
    project.set_source(source=source, pull_at_runtime=True)

    # Set default image:
    if default_image:
        project.set_default_image(default_image)

    if build_image:
        project.build_image(image=f'.{project.name}',
                    base_image='mlrun/mlrun-gpu' if use_gpu else 'mlrun/mlrun',
                    requirements=['tensorflow==2.14.0', 'typing_extensions==4.14.1', 'keras<3.0.0'],
                    overwrite_build_params=True,
                    set_as_default=True,
                    )
        
    # Setting functions
    project.set_function(
        func=os.path.join(framework, "training-and-evaluation.py"),
        name="training-and-evaluation",
        kind="job",
        image=project.default_image
    ).save()
        
    project.set_function(
        func=os.path.join(framework, "serving.py"),
        name="serving", 
        kind="serving", 
        image=project.default_image,
    ).save()

    project.set_function("hub://open_archive", name="open-archive").save()
    
    if not mlrun.mlconf.is_ce_mode():
        project.get_function("training-and-evaluation").apply(mlrun.auto_mount())
        project.get_function("open-archive").apply(mlrun.auto_mount())
        
    onnx_func = project.set_function("hub://onnx_utils", name="onnx-utils")

    # Set the training workflow:
    project.set_workflow("mask_detection_workflow", os.path.join(framework, "workflow.py"), embed=True)

    # Save and return the project:
    project.save()
    return project