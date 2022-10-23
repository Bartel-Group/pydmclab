from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='pydmc',
        version='0.0.1',
        description='facilitating efficient computational materials research',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='https://github.umn.edu/bartel-group/pydmc',
        author=['Bartel Research Group'],
        author_email=['cbartel@umn.edu'],
        license='MIT',
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=[],
        extras_require={},
        classifiers=[],
        test_suite='',
        tests_require=[],
        scripts=[]
    )
    
#        package_data={'modules' : ['module_data/*.json',
#                                   'module_data/*.p']}