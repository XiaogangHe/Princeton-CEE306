# Dive into hydrology using Python 

## Tools you'll need
[Python]: http://www.python.org
[R]: https://www.r-project.org/
[IPython]: http://ipython.org
[Jupyter Notebook]: http://jupyter.org
[NumPy]: http://www.numpy.org
[matplotlib]: http://matplotlib.org
[pandas]: http://pandas.pydata.org/index.html
[rpy2]: https://rpy2.readthedocs.io/en/version_2.8.x/

- [Python]
- [R]
- [IPython] and/or the [Jupyter Notebook]
- Some scientific computing packages:
	- [NumPy]: a general-purpose array-processing package 
    - [matplotlib]: a Python 2D plotting library for data visulization
	- [pandas]: Python's data analysis tools
	- [rpy2]: an interface between Python and R

For **Mac** users, we highly recommend installing [Anaconda](https://www.continuum.io/downloads) (**Python 2.7 version**) as you can use its package manager (called "conda") to easily install and update packages :smile:. For example, open your terminal window and type the following commands to install the above mentioned packages:
- `conda install R`
- `conda install rpy2`

For **Windows** users, you may need to install these packages separately :frowning:. 
- To install rpy2, download the Windows Binaries for rpy2 [here] (http://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2) based on your computer architecture. Launch the Command Prompt and type `pip install SomePackage-1.0-py2.py3-none-any.whl` to install the downloaded .whl files. Please make sure that you have the latest version of pip. You can upgrade pip using `python -m pip install --upgrade pip`.
- Change Path for R. Go to `advanced and system setting` -> `environment variables`
- In the user variable field, add `C:\Program Files\R\R-3.0.2\bin\x64` to the path
- In the system variable field, add three new variables: 
    - Create a `R_HOME` system variable with a value similar to `C:\Program Files\R\R-3.2.0`
    - Create a `R_USER` system variable with your user name `C:\Users\"your user name"`
    - Create a `R_LIBS_USER` system variable with a path to the folder where external R packages are/will be installed. You can find the path through the command `.libPaths()` in the `R` environment.

## Let's get started!
1. Get the GitHub repository
   - Use the command line: `git clone git@github.com:XiaogangHe/Princeton-CEE306.git` 
   - Or download the repository as a .zip file 
2. Run the notebook 
   - Open your terminal and type: `jupyter notebook`

# Resources

## Learn IPython well
*  **[5-10 mins guide to learn IPython Notebook](http://opentechschool.github.io/python-data-intro/core/notebook.html).** (You can [watch the lecture on Youtube](https://www.youtube.com/watch?v=qb7FT68tcA8) instead.)
* [A gallery of interesting Jupyter and IPython Notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-and-IPython-Notebooks)
* [Fabian Pedregosa's gallery](http://nb.bianp.net/sort/views/)

## Learn Pandas well
* **Essential**: [10 Minutes to Pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
* **Essential**: [Things in Pandas I Wish I'd Had Known Earlier](http://nbviewer.ipython.org/github/rasbt/python_reference/blob/master/tutorials/things_in_pandas.ipynb) (IPython Notebook)
* [Useful Pandas Snippets](http://www.swegler.com/becky/blog/2014/08/06/useful-pandas-snippets/)
* Other docs related to pandas:
	* The official [cookbook](http://pandas.pydata.org/pandas-docs/stable/cookbook.html)
	* [Data Structures](http://pandas.pydata.org/pandas-docs/stable/dsintro.html)
	* [Group By: split-apply-combine DataFrames](http://pandas.pydata.org/pandas-docs/stable/groupby.html)
	* [Visulization](http://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html)

## Cheat sheets
Bookmark these cheat sheets:
- [Matplotlib/Pandas/Python cheat sheets](https://drive.google.com/folderview?id=0ByIrJAE4KMTtaGhRcXkxNHhmY2M).

## Questions, answers, chats
Ask questions related to Python on StackExchange [stats.stackexchange.com – _python_.](http://stats.stackexchange.com/questions/tagged/python) 

