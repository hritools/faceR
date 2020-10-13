import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="facer",
    version="0.0.1",
    author="Ruslan",
    author_email="r.gayfullin@innopolis.ru",
    description="A package for face recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://cordelianew.university.innopolis.ru/gitea/gruslan/faceR",
    packages=setuptools.find_packages(),
    package_data={
        'faceR': ['config.yaml', 'logging.yaml']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'PyYAML',
        'pyzmq',
        'numpy',
        'tensorflow',
        'scipy',
        'scikit-learn',
        'opencv-python',
        'imageio',
        'Pillow',
        'dlib'
    ],
)
