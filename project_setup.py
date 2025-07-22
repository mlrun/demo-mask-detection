import mlrun

def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    
    # Save and return the project:
    project.save()
    return project