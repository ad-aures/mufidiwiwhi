from setuptools import setup
setup(
    name = 'mufidiwiwhi',
    version = '0.0.1',
    packages = ['mufidiwiwhi'],
    entry_points = {
        'console_scripts': [
            'mufidiwiwhi = mufidiwiwhi.transcribe:cli'
        ]
    })
