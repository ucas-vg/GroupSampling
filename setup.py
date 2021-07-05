from setuptools import setup, find_packages


setup(name='GroupSampling',
      version='1.0.0',
      description='Group Sampling for Unsupervised Person Re-identification',
      author='wavinflaghxm',
      author_email='',
      url='https://github.com/wavinflaghxm/GroupSampling',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Person Re-identification'
      ])
