from glob import glob

from setuptools import setup

package_name = "ucslam_px4"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anonymous",
    maintainer_email="anonymous@example.com",
    description="Bridge from UC-SLAM to PX4 through MAVROS",
    license="GPL-3.0-or-later",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["px4_bridge = ucslam_px4.px4_bridge:main"]},
)
