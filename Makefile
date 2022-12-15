install:
	@pip install -e .
# @pip install . # for procduction

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@rm -Rf */*egg-info

all: install clean

run_train:
	python -c 'from modules.interface.main import train; train()'

run_api:
	@python modules/api/fast_api.py

test:
	@pytest -v tests


#################### TESTS #####################

test_cloud_training: test_gcp_setup test_gcp_project test_gcp_bucket test_big_query test_cloud_data

test_gcp_setup:
	@TEST_ENV=development pytest \
	tests/setup/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/setup/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/setup/test_gcp_setup.py::TestGcpSetup::test_code_get_project

test_gcp_project:
	@TEST_ENV=development pytest \
	tests/setup/test_gcp_setup.py::TestGcpSetup::test_setup_project_id

test_gcp_bucket:
	@TEST_ENV=development pytest \
	tests/setup/test_gcp_setup.py::TestGcpSetup::test_setup_bucket_exists \
	tests/setup/test_gcp_setup.py::TestGcpSetup::test_setup_bucket_name

test_big_query:
	@TEST_ENV=development pytest \
	tests/cloud_data/test_cloud_data.py::TestCloudData::test_big_query_dataset_variable_exists \
	tests/cloud_data/test_cloud_data.py::TestCloudData::test_cloud_data_create_dataset \
	tests/cloud_data/test_cloud_data.py::TestCloudData::test_cloud_data_create_table \
	tests/cloud_data/test_cloud_data.py::TestCloudData::test_cloud_data_table_content

test_cloud_data:
	@TEST_ENV=development pytest tests/cloud_data/test_cloud_data.py::TestCloudData::test_cloud_data_bq_chunks
