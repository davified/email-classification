# Machine learning project template

## Getting started

1. Fork the repo

2. Clone the repo and `cd` into the directory: `git clone https://github.com/your-username/tw-ml-template your-project-name && cd your-project-name`

3. Run `bin/setup.sh`. This script will:

	- Install python3

	- Create a virtual environment folder in your project directory and install some dependencies which are commonly used in machine learning, such as:
		- jupyter
		- pandas
		- numpy
		- sklearn
		- nose (for unit tests)

4. To activate the virtual environment `source .venv/bin/activate`

5. To run tests: `bin/run_tests.sh`

6. To work with jupyter notebook, run `jupyter notebook`
