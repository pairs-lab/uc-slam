from setuptools import setup

package_name = "ucslam_depth"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anonymous",
    maintainer_email="anonymous@example.com",
    description="UC-SLAM depth node for ROS 2",
    license="GPL-3.0-or-later",
    entry_points={"console_scripts": ["depth_node = ucslam_depth.depth_node:main"]},
)
