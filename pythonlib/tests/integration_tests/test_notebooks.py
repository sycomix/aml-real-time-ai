# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License
import pytest
import os
import nbformat
import json

from nbconvert.preprocessors import ExecutePreprocessor
from tests.integration_tests.test_utils import get_service_principal, get_test_config, download_kaggle_test_data

def test_quickstart_notebook():
    run_notebook('notebooks/resnet50/00_QuickStart.ipynb')

def test_modelbuild_notebook():
    download_kaggle_test_data()
    run_notebook('notebooks/resnet50/01_ModelBuild.ipynb')

def run_notebook(path):
    file = os.path.join(os.getcwd(), path)
    lines = None
    with open(file, encoding="utf-8") as f:
        lines = f.read()
        f.close()
    
    lines = replace_auth_values(lines)
    nb = nbformat.reads(lines, as_version=4)
    ep = ExecutePreprocessor(timeout=1800, kernel_name='python3')
    out = ep.preprocess(nb, {'metadata': {'path': 'notebooks/resnet50'}})
    print(out)

def replace_auth_values(str):
    test_config = get_test_config()
    service_principal = get_service_principal()

    str = str.replace(
        'deployment_client = DeploymentClient(subscription_id, resource_group, model_management_account)',
        f"""from collections import namedtuple\\nAad_Record = namedtuple(\\"AadRecord\\", [\\"tenant\\",\\"service_principal_id\\",\\"service_principal_key\\"])\\nservice_principal=Aad_Record(\\"{service_principal.tenant}\\", \\"{service_principal.service_principal_id}\\", \\"{service_principal.service_principal_key}\\")\\ndeployment_client = DeploymentClient(\\"{test_config['test_subscription_id']}\\", \\"{test_config['test_resource_group']}\\", \\"{test_config['test_model_management_account']}\\", service_principal)""",
    )

    str = str.replace(
        'featurizer = RemoteQuantizedResNet50(subscription_id, resource_group, model_management_account, model_path)',
        f"""import uuid\\nfrom collections import namedtuple\\nAad_Record = namedtuple(\\"AadRecord\\", [\\"tenant\\",\\"service_principal_id\\",\\"service_principal_key\\"])\\nservice_principal=Aad_Record(\\"{service_principal.tenant}\\", \\"{service_principal.service_principal_id}\\", \\"{service_principal.service_principal_key}\\")\\nremote_service_name = (\\"int-test-featurizer-svc-\\" + str(uuid.uuid4()))[:30]\\nfeaturizer = RemoteQuantizedResNet50(\\"{test_config['test_subscription_id']}\\", \\"{test_config['test_resource_group']}\\", \\"{test_config['test_model_management_account']}\\", model_path, remote_service_name, service_principal_params=service_principal)""",
    )

    str = str.replace(
        'service_name = \"quickstart-master-service\"', 
        'import uuid\\nservice_name = (\\"int-test-qs-svc-\\" + str(uuid.uuid4()))[:30]')

    str = str.replace(
        'service_name = \"modelbuild-service\"', 
        'import uuid\\nservice_name = (\\"int-test-mb-svc-\\" + str(uuid.uuid4()))[:30]')

    return str