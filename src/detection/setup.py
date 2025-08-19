from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/prompts', glob('prompts/*.txt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Detection package for food and face detection',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'food_detection_service_node = detection.food_detection_service_node:main',
            'face_detection_service_node = detection.face_detection_service_node:main',
        ],
    },
)