import os

import boto3
import pytest
import requests

"""
Make sure env variable AWS_SAM_STACK_NAME exists with the name of the stack we are going to test. 
"""


class TestApiGateway:

    @pytest.fixture()
    def stack_outputs(self):
        """ Get the Cloudformation Stack outputs """
        stack_name = os.environ.get("AWS_SAM_STACK_NAME")

        if stack_name is None:
            raise ValueError('Please set the AWS_SAM_STACK_NAME environment variable to the name of your stack')

        client = boto3.client("cloudformation")

        try:
            response = client.describe_stacks(StackName=stack_name)
        except Exception as e:
            raise Exception(
                f"Cannot find stack {stack_name} \n" f'Please make sure a stack with the name "{stack_name}" exists'
            ) from e

        stacks = response["Stacks"]
        return {output["OutputKey"]: output["OutputValue"] for output in stacks[0]["Outputs"]}

    @pytest.fixture()
    def api_gateway_url(self, stack_outputs):
        """ Get the API Gateway URL from Cloudformation Stack outputs """
        return stack_outputs["HelloWorldApi"]

    @pytest.fixture()
    def api_key(self, stack_outputs):
        """ Resolve the API key value for the ApiKeyId stack output """
        client = boto3.client("apigateway")
        return client.get_api_key(apiKey=stack_outputs["ApiKeyId"], includeValue=True)["value"]

    def test_api_gateway_requires_api_key(self, api_gateway_url):
        """ Calling the API Gateway endpoint without an API key should be rejected """
        response = requests.get(api_gateway_url)

        assert response.status_code == 403

    def test_api_gateway(self, api_gateway_url, api_key):
        """ Call the API Gateway endpoint with a valid API key and check the response """
        response = requests.get(api_gateway_url, headers={"x-api-key": api_key})

        assert response.status_code == 200
        assert response.json() == {"message": "hello world"}
