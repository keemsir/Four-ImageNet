from setuptools import setup, find_namespace_packages

setup(
    name='fournet',
    packages=find_namespace_packages(where='module'),
    version='1.0.0',
    url='https://keemsir.github.io/',
    author='Meangee.Keem',
    author_email='keemsir@gmail.com',
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "matplotlib"
        # ,"util_msg"
    ],
    entry_points={
        'console_scripts': [
            'fournet_train = module.training:main',
            'fournet_predict = module.predict:main'
        ],
    }
)
