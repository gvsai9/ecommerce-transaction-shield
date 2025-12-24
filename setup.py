from setuptools import setup, find_packages


def get_requirements(file_path="requirements.txt"):
    """
    This function returns the list of requirements
    from the requirements.txt file
    """
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="ecommerce_transaction_shield",
    version="0.1.0",
    author="Venkata Sai",
    description="End-to-end ML pipeline with MLOps practices",
    packages=find_packages(),
    install_requires=get_requirements(),
)
